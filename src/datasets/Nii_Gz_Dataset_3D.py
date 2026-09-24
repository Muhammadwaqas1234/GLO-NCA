from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    from scipy.ndimage import map_coordinates as _map_coordinates
    _NDIMAGE = True
except Exception:  # heavy-aug elastic/blur fall back to no-op if scipy missing
    _NDIMAGE = False


class Dataset_NiiGz_3D_BraTS(Dataset_3D):
    r"""3D loader for multi-modal BraTS: one folder per case with four modalities and a seg.

    Modalities are stacked as 4 channels (T1, T1ce, T2, FLAIR). Raw labels (1 NCR, 2 ED,
    4 ET, sometimes remapped to 3) become nested regions: WT {1,2,4} -> ch 0,
    TC {1,4} -> ch 1, ET {4} -> ch 2. 3D only (``slice`` must be None).
    """

    # Modality suffixes in channel order T1, T1ce, T2, FLAIR (BraTS 2024 naming);
    # override MODALITIES for the older t1/t1ce/t2/flair names.
    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
    SEG_SUFFIX = "seg"

    def getFilesInPath(self, path):
        r"""Discover cases recursively: a case folder directly holds four modalities + seg.

        folder_name is relative to ``path`` (handles nested cohorts); the case id is the leaf folder.
        Returns {caseID: {0: (rel_path, caseID, 0)}}.
        """
        from src.experiment.datasource import discover_cases
        dic = {}
        for case_id, rel_path in discover_cases(path):
            dic[case_id] = {0: (rel_path, case_id, 0)}
        return dic

    def _find_modality_file(self, folder, patient, suffix):
        r"""Find a modality/seg file in a case folder, tolerant to naming (e.g. suffix 't1c' or 'seg')."""
        # Explicit names, '_' or '-' separator, .nii.gz or .nii.
        for sep in ("_", "-"):
            for ext in (".nii.gz", ".nii"):
                candidate = os.path.join(folder, f"{patient}{sep}{suffix}{ext}")
                if os.path.exists(candidate):
                    return candidate
        # Fallback: any file ending with that modality suffix, either separator.
        endings = (f"_{suffix}.nii.gz", f"_{suffix}.nii",
                   f"-{suffix}.nii.gz", f"-{suffix}.nii")
        for f in os.listdir(folder):
            if f.lower().endswith(endings):
                return os.path.join(folder, f)
        raise FileNotFoundError(f"Could not find '{suffix}' volume for patient '{patient}' in {folder}")

    def load_item(self, path):
        r"""Load one nii/nii.gz volume as float; header open and decompression are profiled separately."""
        from src.profiling import get_profiler
        prof = get_profiler()
        if not prof.enabled:
            return nib.load(path).get_fdata()

        with prof.section("data/nifti_header"):
            handle = nib.load(path)
        try:
            prof.record("data/file_size_mb", os.path.getsize(path) / 1024 ** 2)
        except OSError:
            pass
        with prof.section("data/nifti_materialize"):
            return handle.get_fdata()

    @staticmethod
    def _foreground_bbox(vol_stack):
        r"""Bounding box (x0,x1,y0,y1,z0,z1) of non-zero brain across modalities, or None if empty."""
        fg = np.any(vol_stack > 0, axis=-1)
        if not fg.any():
            return None
        xs = np.where(fg.any(axis=(1, 2)))[0]
        ys = np.where(fg.any(axis=(0, 2)))[0]
        zs = np.where(fg.any(axis=(0, 1)))[0]
        return xs[0], xs[-1] + 1, ys[0], ys[-1] + 1, zs[0], zs[-1] + 1

    def _labels_to_regions(self, seg):
        r"""Raw BraTS labels {0..4} -> (X, Y, Z, 3) binary WT, TC, ET."""
        # ET is 4, or 3 in some remapped copies.
        et = np.logical_or(seg == 4, seg == 3)
        ncr = (seg == 1)
        ed = (seg == 2)

        wt = np.logical_or(np.logical_or(ncr, ed), et)   # whole tumor
        tc = np.logical_or(ncr, et)                       # tumor core

        label = np.stack([wt, tc, et], axis=-1).astype(np.float32)
        return label

    # Base seed for per-(epoch, case) augmentation; set by the runner.
    _aug_base_seed = None

    # Optional training patch smaller than the working volume (off in production).
    # self.size is the 128³ working volume (resample target); None keeps patch == working
    # volume. Train only: validation/test geometry is unaffected.
    _train_patch_size = None

    def set_train_patch_size(self, size):
        """Set a training patch smaller than the working volume (int or 3-sequence); None disables it."""
        if size is None:
            self._train_patch_size = None
            return
        if isinstance(size, int):
            size = (size, size, size)
        self._train_patch_size = tuple(int(v) for v in size)

    # Optional deterministic preprocessing cache; None disables it.
    _precache = None

    def set_preprocess_cache(self, cache):
        """Attach a PreprocessCache (deterministic head only; augmentation runs every epoch)."""
        self._precache = cache

    def set_augmentation_seed(self, seed):
        r"""Set the base seed for per-(epoch, case) augmentation RNG."""
        self._aug_base_seed = int(seed)

    def __getitem__(self, idx):
        r"""Load and preprocess one case.

        #Args
            idx: index, or (epoch, index) from the runner's sampler.
        #Returns:
            id (str): '_<patient>_0'
            img: (X, Y, Z, 4) float32, T1/T1ce/T2/FLAIR
            label: (X, Y, Z, 3) float32, WT/TC/ET
        """
        # Seed augmentation from (base_seed, epoch, index): different every epoch, yet
        # deterministic and independent of worker count and order.
        epoch = None
        if isinstance(idx, (tuple, list)) and len(idx) == 2:
            epoch, idx = int(idx[0]), int(idx[1])
        if epoch is not None and self._aug_base_seed is not None:
            s = (self._aug_base_seed * 1_000_003 + epoch * 9_176_231 + idx) % (2 ** 32)
            random.seed(s)
            np.random.seed(s)

        # Profiler is a no-op unless enabled.
        from src.profiling import get_profiler
        _prof = get_profiler()

        crop_fg = self.exp.get_from_config('foreground_crop') is True

        key = self.images_list[idx]
        cached = self.data.get_data(key=key)
        # Cache lookup consumes no random numbers, so hit or miss leaves the RNG unchanged.
        if not cached and self._precache is not None:
            folder_name, p_id, _ = key
            _hit = self._precache.get(str(p_id))
            if _hit is not None:
                with _prof.section("data/cache_hit"):
                    _img, _label = _hit
                cached = ("_" + str(p_id) + "_0", _img, _label)
                self.data.set_data(key=key, data=cached)

        if not cached:
            folder_name, p_id, _ = key
            folder = os.path.join(self.images_path, folder_name)

            # Load raw modalities + seg.
            raw_vols = [self.load_item(self._find_modality_file(folder, folder_name, mod))
                        for mod in self.MODALITIES]
            raw = np.stack(raw_vols, axis=-1)  # (X, Y, Z, C) full resolution
            seg = self.load_item(self._find_modality_file(folder, folder_name, self.SEG_SUFFIX))

            # Crop to the brain bounding box.
            with _prof.section("data/foreground_crop"):
                if crop_fg:
                    bbox = self._foreground_bbox(raw)
                    if bbox is not None:
                        x0, x1, y0, y1, z0, z1 = bbox
                        raw = raw[x0:x1, y0:y1, z0:z1, :]
                        seg = seg[x0:x1, y0:y1, z0:z1]

            # Resize to the working volume.
            with _prof.section("data/resample"):
                if self.exp.get_from_config('rescale') is not False:
                    img = np.stack([self.rescale3d(raw[..., c]) for c in range(raw.shape[-1])], axis=-1)
                    seg = self.rescale3d(seg, isLabel=True)
                else:
                    img = raw
            with _prof.section("data/labels_to_regions"):
                label = self._labels_to_regions(seg)  # (X, Y, Z, 3)

            img_id = "_" + str(p_id) + "_0"
            # Cache the deterministic result only.
            if self._precache is not None:
                with _prof.section("data/cache_write"):
                    self._precache.put(str(p_id), img, label)
            self.data.set_data(key=key, data=(img_id, img, label))
            cached = self.data.get_data(key=key)

        img_id, img, label = cached

        # Optional training patch (inert unless configured).
        if self.exp.get_from_config('patchify') is True and self.state == "train":
            with _prof.section("data/patchify"):
                img, label = self.patchify_multimodal(img, label)

        # Train-only augmentation, gated by the 'augment' config flag.
        if self.exp.get_from_config('augment') is True and self.state == "train":
            with _prof.section("data/augment"):
                img, label = self._augment(img, label)

        # Per-modality intensity normalisation.
        if self.exp.get_from_config('nonzero_norm') is True:
            # Non-zero z-norm over brain voxels only.
            img_norm = np.empty_like(img, dtype=np.float32)
            for c in range(img.shape[-1]):
                ch = img[..., c]
                mask = ch > 0
                if mask.sum() > 0:
                    img_norm[..., c] = np.where(
                        mask, (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8), 0.0)
                else:
                    img_norm[..., c] = ch
        else:
            # Fallback: torchio z-norm + rescale to [0, 1] (production uses non-zero z-norm).
            rescale = torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
            znormalisation = torchio.ZNormalization()
            img_norm = np.empty_like(img, dtype=np.float32)
            for c in range(img.shape[-1]):
                channel = np.expand_dims(img[..., c], axis=0)
                if np.sum(channel) > 0:
                    channel = znormalisation(channel)
                channel = rescale(channel)
                img_norm[..., c] = channel[0]
        img = img_norm

        return (img_id, img, label)

    def rescale3d(self, img, isLabel=False):
        r"""Resize a volume to the working size: linear for images (no cubic overshoot), nearest for labels."""
        size = (self.size[0], self.size[1])
        size2 = (self.size[2], self.size[0])
        interp = cv2.INTER_NEAREST if isLabel else cv2.INTER_LINEAR

        resized = np.zeros((self.size[0], self.size[1], img.shape[2]), dtype=np.float32)
        for z in range(img.shape[2]):
            resized[:, :, z] = cv2.resize(img[:, :, z], dsize=size, interpolation=interp)

        if len(self.size) == 3:
            tmp = resized
            resized = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=np.float32)
            for y in range(tmp.shape[1]):
                resized[:, y, :] = cv2.resize(tmp[:, y, :], dsize=size2, interpolation=interp)
        return resized

    def _augment(self, img, label):
        r"""Label-safe augmentation for image (X,Y,Z,C) and label (X,Y,Z,R).

        'light': flips, 90-degree rotations, per-modality scale/shift.
        'heavy': light + gamma, noise, blur and elastic deformation.
        Geometric transforms share axes/fields with the label; intensity transforms touch the
        image only and keep zeros at zero.
        """
        level = self.exp.get_from_config('augment_level') or 'light'

        # Flips.
        for ax in (0, 1, 2):
            if random.random() < 0.5:
                img = np.flip(img, axis=ax)
                label = np.flip(label, axis=ax)
        # 90-degree axial rotation.
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k, axes=(0, 1))
            label = np.rot90(label, k, axes=(0, 1))
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)

        # Heavy: elastic deformation (one field; nearest for labels).
        if level == 'heavy' and random.random() < 0.3:
            img, label = self._elastic(img, label)

        # Per-modality scale + shift (image only).
        img = img.copy()
        s_range = 0.1 if level != 'heavy' else 0.2
        for c in range(img.shape[-1]):
            ch = img[..., c]
            mask = ch != 0
            if not mask.any():
                continue
            scale = 1.0 + random.uniform(-s_range, s_range)
            shift = random.uniform(-s_range, s_range) * (ch[mask].std() + 1e-8)
            ch[mask] = ch[mask] * scale + shift
            if level == 'heavy':
                # Gamma (contrast) on the min-max normalised brain voxels.
                if random.random() < 0.3:
                    v = ch[mask]
                    lo, hi = v.min(), v.max()
                    if hi > lo:
                        g = random.uniform(0.7, 1.5)
                        ch[mask] = ((v - lo) / (hi - lo)) ** g * (hi - lo) + lo
                # Additive Gaussian noise.
                if random.random() < 0.2:
                    ch[mask] = ch[mask] + np.random.normal(
                        0, 0.05 * (ch[mask].std() + 1e-8), size=ch[mask].shape)
                # Gaussian blur (whole channel; zeros stay ~zero).
                if random.random() < 0.2 and _NDIMAGE:
                    img[..., c] = _gaussian_filter(img[..., c], sigma=random.uniform(0.4, 0.8))

        return np.ascontiguousarray(img), np.ascontiguousarray(label)

    def _elastic(self, img, label, alpha=8.0, sigma=3.0):
        r"""Elastic deformation shared by image (linear) and label (nearest); skipped without scipy."""
        if not _NDIMAGE:
            return img, label
        shape = img.shape[:3]
        # One smooth random displacement field, shared by image and label.
        dx = _gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = _gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = _gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        gx, gy, gz = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                                 np.arange(shape[2]), indexing='ij')
        coords = [np.reshape(gx + dx, -1), np.reshape(gy + dy, -1), np.reshape(gz + dz, -1)]
        out_img = np.empty_like(img)
        for c in range(img.shape[-1]):
            out_img[..., c] = _map_coordinates(
                img[..., c], coords, order=1, mode='nearest').reshape(shape)
        out_lab = np.empty_like(label)
        for r in range(label.shape[-1]):
            out_lab[..., r] = _map_coordinates(
                label[..., r], coords, order=0, mode='nearest').reshape(shape)
        return out_img, out_lab

    def patchify_multimodal(self, img, label):
        r"""Random patch shared across channels, optionally biased toward a tumour region."""
        # self.size is the working volume; a configured training patch makes this a real crop.
        size = self._train_patch_size or self.size
        prioritize = self.exp.get_from_config('priotize_masks')
        contains_mask = prioritize is not None and (random.uniform(0, 1) < prioritize)
        # Region to bias toward: 0=WT (default), 1=TC, 2=ET.
        region = self.exp.get_from_config('prioritize_region')
        region = 0 if region is None else int(region)

        pos_x = pos_y = pos_z = 0
        fallback = None  # best WT-containing position, fallback if the target region is never found

        # When the volume already equals the patch size every retry sees the same region,
        # so the two reductions are hoisted; the loop and all random.* calls are unchanged.
        full_volume = tuple(img.shape[:3]) == tuple(size)
        if full_volume and contains_mask:
            region_present = bool(label[..., region].max() > 0)
            wt_present = bool(label[..., 0].max() > 0) if region != 0 else False

        for _ in range(50):  # bounded retries to find a region-containing patch
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])
            if not contains_mask:
                break
            if full_volume:
                # Position is always (0,0,0); the verdict cannot change.
                if region_present:
                    break
                if region != 0 and fallback is None and wt_present:
                    fallback = (0, 0, 0)
                continue
            patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], region]
            if patch.max() > 0:
                break  # found a patch containing the target region
            # First WT-valid position is the fallback (ET may be tiny or absent).
            if region != 0 and fallback is None:
                wt_patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0]
                if wt_patch.max() > 0:
                    fallback = (pos_x, pos_y, pos_z)
        else:
            # Retries exhausted: use the WT-valid fallback.
            if fallback is not None:
                pos_x, pos_y, pos_z = fallback

        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        return img, label
