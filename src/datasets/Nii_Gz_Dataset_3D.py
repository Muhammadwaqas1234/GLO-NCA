from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio


class Dataset_NiiGz_3D_BraTS(Dataset_3D):
    r"""3D loader for the multi-modal BraTS dataset (Kaggle / official layout).

    Each patient lives in its own folder containing four modality volumes and a
    segmentation mask::

        BraTS20_Training_001/
            BraTS20_Training_001_t1.nii.gz
            BraTS20_Training_001_t1ce.nii.gz
            BraTS20_Training_001_t2.nii.gz
            BraTS20_Training_001_flair.nii.gz
            BraTS20_Training_001_seg.nii.gz

    The four modalities are stacked into a 4-channel input (T1, T1ce, T2,
    FLAIR). The raw label values (1 = NCR, 2 = ED, 4 = ET; some Kaggle copies
    remap 4 -> 3) are converted into the three standard, nested BraTS regions
    used for reporting:

        WT (Whole Tumor)      = labels {1, 2, 4}   -> channel 0
        TC (Tumor Core)       = labels {1, 4}      -> channel 1
        ET (Enhancing Tumor)  = label  {4}         -> channel 2

    Only the 3D path is supported (``slice`` must be None); BraTS volumes are
    segmented as full 3D volumes.
    """

    # Modality suffixes in the fixed channel order T1, T1ce, T2, FLAIR.
    # BraTS 2024 (BraTS-GLI) uses t1n / t1c / t2w / t2f; override MODALITIES on
    # the instance if your dataset uses the older t1/t1ce/t2/flair names.
    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
    SEG_SUFFIX = "seg"

    def getFilesInPath(self, path):
        r"""Discover patients by folder. The 'images' and 'labels' live in the
            same per-patient folder, so both image_path and label_path point to
            the BraTS root.
            #Args
                path (string): BraTS root directory (one sub-folder per patient)
            #Returns:
                dic (dictionary): {patientID: {0: (folder_name, patientID, 0)}}
        """
        dic = {}
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if not os.path.isdir(full):
                continue
            dic[entry] = {0: (entry, entry, 0)}
        return dic

    def _find_modality_file(self, folder, patient, suffix):
        r"""Locate a modality/seg file in a patient folder, tolerant to naming.
            #Args
                folder (str): absolute path to the patient folder
                patient (str): patient id (folder name)
                suffix (str): modality suffix, e.g. 't1ce' or 'seg'
        """
        # Common explicit names (underscore or hyphen separator, .nii.gz/.nii).
        # BraTS 2020: 'BraTS_x_t1.nii.gz'; BraTS 2024: 'BraTS-GLI-x-t1c.nii'.
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
        r"""Load a single nii/nii.gz volume as a float numpy array."""
        return nib.load(path).get_fdata()

    @staticmethod
    def _foreground_bbox(vol_stack):
        r"""Bounding box of the non-zero brain across all modalities.

            Swin-UNETR-style CropForeground: removes the black background so the
            resized patch contains brain only (recovers small ET/TC detail).
            #Args
                vol_stack (numpy): (X, Y, Z, C) stacked modalities
            #Returns
                (x0,x1,y0,y1,z0,z1) or None if the volume is empty
        """
        fg = np.any(vol_stack > 0, axis=-1)
        if not fg.any():
            return None
        xs = np.where(fg.any(axis=(1, 2)))[0]
        ys = np.where(fg.any(axis=(0, 2)))[0]
        zs = np.where(fg.any(axis=(0, 1)))[0]
        return xs[0], xs[-1] + 1, ys[0], ys[-1] + 1, zs[0], zs[-1] + 1

    def _labels_to_regions(self, seg):
        r"""Convert raw BraTS segmentation values into nested ET/TC/WT regions.
            #Args
                seg (numpy): raw label volume with values in {0,1,2,3,4}
            #Returns:
                label (numpy): (X, Y, Z, 3) binary volume, channels = WT, TC, ET
        """
        # ET is encoded as 4 in BraTS2020 and sometimes remapped to 3 on Kaggle.
        et = np.logical_or(seg == 4, seg == 3)
        ncr = (seg == 1)
        ed = (seg == 2)

        wt = np.logical_or(np.logical_or(ncr, ed), et)   # whole tumor
        tc = np.logical_or(ncr, et)                       # tumor core

        label = np.stack([wt, tc, et], axis=-1).astype(np.float32)
        return label

    def __getitem__(self, idx):
        r"""Load and preprocess one BraTS patient.
            #Returns:
                id (str): patient identifier, formatted '_<patient>_0'
                img (numpy): (X, Y, Z, 4) float32, modalities T1/T1ce/T2/FLAIR
                label (numpy): (X, Y, Z, 3) float32, regions WT/TC/ET
        """
        rescale = torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
        znormalisation = torchio.ZNormalization()

        crop_fg = self.exp.get_from_config('foreground_crop') is True

        key = self.images_list[idx]
        cached = self.data.get_data(key=key)
        if not cached:
            folder_name, p_id, _ = key
            folder = os.path.join(self.images_path, folder_name)

            # --- Load RAW modalities + seg (no resize yet if we crop first) --
            raw_vols = [self.load_item(self._find_modality_file(folder, folder_name, mod))
                        for mod in self.MODALITIES]
            raw = np.stack(raw_vols, axis=-1)  # (X, Y, Z, C) full resolution
            seg = self.load_item(self._find_modality_file(folder, folder_name, self.SEG_SUFFIX))

            # --- Foreground crop to the brain bounding box (Swin-UNETR) ------
            if crop_fg:
                bbox = self._foreground_bbox(raw)
                if bbox is not None:
                    x0, x1, y0, y1, z0, z1 = bbox
                    raw = raw[x0:x1, y0:y1, z0:z1, :]
                    seg = seg[x0:x1, y0:y1, z0:z1]

            # --- Resize to training size -----------------------------------
            if self.exp.get_from_config('rescale') is not False:
                img = np.stack([self.rescale3d(raw[..., c]) for c in range(raw.shape[-1])], axis=-1)
                seg = self.rescale3d(seg, isLabel=True)
            else:
                img = raw
            label = self._labels_to_regions(seg)  # (X, Y, Z, 3)

            img_id = "_" + str(p_id) + "_0"
            self.data.set_data(key=key, data=(img_id, img, label))
            cached = self.data.get_data(key=key)

        img_id, img, label = cached

        # Patchify on the fly for training (global info comes from the
        # coarse NCA level, so a patch is enough at full resolution).
        if self.exp.get_from_config('patchify') is True and self.state == "train":
            img, label = self.patchify_multimodal(img, label)

        # Light on-the-fly augmentation (train only). Gated by the 'augment'
        # config flag -- default off, so single-modality / older configs are
        # unchanged. Cheap, label-safe geometric + intensity transforms that
        # regularise without distorting tumour shape: axis flips, 90-deg
        # in-plane rotations and a small per-modality intensity scale/shift.
        if self.exp.get_from_config('augment') is True and self.state == "train":
            img, label = self._augment(img, label)

        # Per-modality intensity normalisation.
        if self.exp.get_from_config('nonzero_norm') is True:
            # Swin-UNETR nonzero z-norm: normalise using brain voxels only.
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
            # Original torchio z-norm + rescale-to-[0,1] per channel.
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
        r"""Resize a 3D volume to the configured training size (X, Y, Z).

            Images use LINEAR interpolation and labels NEAREST. Linear is the
            standard for medical volumes: cubic overshoots at brain/tumour edges,
            producing negative intensities and ringing right where the small
            ET/TC structures live, which corrupts the smallest regions. Nearest
            for labels keeps the mask strictly binary.
        """
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
        r"""Label-safe augmentation for a (X,Y,Z,C) image and (X,Y,Z,R) label.

        Geometric transforms are applied identically to image and label so the
        masks stay aligned; intensity transforms touch the image only. All are
        volume-preserving (no interpolation of the label), which keeps the small
        ET/TC regions intact.
            #Args
                img (numpy): (X, Y, Z, C) patch
                label (numpy): (X, Y, Z, R) patch, same spatial dims
            #Returns
                img, label: augmented arrays, same shapes
        """
        # Random flips along each spatial axis (X, Y, Z).
        for ax in (0, 1, 2):
            if random.random() < 0.5:
                img = np.flip(img, axis=ax)
                label = np.flip(label, axis=ax)
        # Random 90-degree rotation in the axial (X, Y) plane (k * 90 deg).
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k, axes=(0, 1))
            label = np.rot90(label, k, axes=(0, 1))
        # Small per-modality intensity scale + shift (image only). Applied to
        # brain voxels; keeps zeros as zeros so the non-zero norm is unaffected.
        img = img.copy()
        for c in range(img.shape[-1]):
            ch = img[..., c]
            mask = ch != 0
            if mask.any():
                scale = 1.0 + random.uniform(-0.1, 0.1)
                shift = random.uniform(-0.1, 0.1) * (ch[mask].std() + 1e-8)
                ch[mask] = ch[mask] * scale + shift
        # np.flip / np.rot90 return views; return contiguous copies so downstream
        # torch.from_numpy does not choke on negative strides.
        return np.ascontiguousarray(img), np.ascontiguousarray(label)

    def patchify_multimodal(self, img, label):
        r"""Random 3D patch of the configured size, shared across all channels.
            Optionally biased towards patches containing tumour (WT channel).
            #Args
                img (numpy): (X, Y, Z, 4)
                label (numpy): (X, Y, Z, 3)
        """
        size = self.size
        prioritize = self.exp.get_from_config('priotize_masks')
        contains_mask = prioritize is not None and (random.uniform(0, 1) < prioritize)
        # Which region to bias the patch toward: 0=WT (default), 1=TC, 2=ET.
        # Biasing toward ET (the rarest region) improves ET/TC recall.
        region = self.exp.get_from_config('prioritize_region')
        region = 0 if region is None else int(region)

        pos_x = pos_y = pos_z = 0
        fallback = None  # best WT-containing position, used if the target region
        #                  (e.g. ET) is never found within the retry budget.
        for _ in range(50):  # bounded retries to find a region-containing patch
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])
            if not contains_mask:
                break
            patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], region]
            if patch.max() > 0:
                break  # found a patch containing the target region
            # Remember the first WT-valid position as a fallback (ET can be tiny
            # or absent in a given patient, so the target region may not exist).
            if region != 0 and fallback is None:
                wt_patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0]
                if wt_patch.max() > 0:
                    fallback = (pos_x, pos_y, pos_z)
        else:
            # Retry budget exhausted without hitting the target region: use the
            # remembered WT-valid patch rather than the last (possibly empty) one.
            if fallback is not None:
                pos_x, pos_y, pos_z = fallback

        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        return img, label
