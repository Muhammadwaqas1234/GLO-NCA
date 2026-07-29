from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio

class Dataset_NiiGz_3D(Dataset_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """

    def getDataShapes():
        return

    def getFilesInPath(self, path):
        r"""Get files in path ordered by id and slice
            #Args
                path (string): The path which should be worked through
            #Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = os.listdir(os.path.join(path))
        dic = {}
        for id_f, f in enumerate(dir_files):
            id = f
            # 2D 
            if self.slice is not None:
                for slice in range(self.getSlicesOnAxis(os.path.join(path, f), self.slice)):
                    if id not in dic:
                        dic[id] = {}
                    dic[id][slice] = (f, id_f, slice)
            # 3D
            else:
                if id not in dic:
                    dic[id] = {}
                dic[id][0] = (f, f, 0)           
        return dic

    def getSlicesOnAxis(self, path, axis):
        return self.load_item(path).shape[axis]

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()

    def rotate_image(self, image, angle, label = False):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        if label:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
        else:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def preprocessing3d(self, img, isLabel=False):
        r"""Preprocess data to fit the required shape
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if not isLabel:
            # TODO: Currently only single volume, no multi phase
            if len(img.shape) == 4:
                img = img[..., 0]
            padded = np.zeros(self.size)#np.random.rand(*self.size) * 0.01
        else:
            padded = np.zeros(self.size)
        img_shape = img.shape
        padded[0:img_shape[0], 0:img_shape[1], 0:img_shape[2]] = img

        return padded

    def rescale3d(self, img, isLabel=False):
        r"""Rescale input image to fit training size
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if len(self.size) == 3:
            size = (self.size[0], self.size[1])
            size2 = (self.size[2], self.size[0])
        else:
            size = (self.size[0], self.size[1])

        img_resized = np.zeros((self.size[0], self.size[1], img.shape[2])) 
        for x in range(img.shape[2]):
            if not isLabel:
                img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=size, interpolation=cv2.INTER_CUBIC) 
            else:
                img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=size, interpolation=cv2.INTER_NEAREST) 

        if len(self.size) == 3 and True:
            img = img_resized
            img_resized = np.zeros((self.size[0], self.size[1], self.size[2]))
            for x in range(img.shape[1]):
                if not isLabel:
                    img_resized[:, x, :] = cv2.resize(img[:, x, :], dsize=size2, interpolation=cv2.INTER_CUBIC) 
                else:
                    img_resized[:, x, :] = cv2.resize(img[:, x, :], dsize=size2, interpolation=cv2.INTER_NEAREST) 

        return img_resized

    def patchify(self, img, label):
        r"""Take a patch of the input. This should be used instead of rescaling if global information is not required.
            #Args
                img (numpy): Image data
                label (numpy): Label data
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        size = self.size

        containsMask = (random.uniform(0, 1) < self.exp.get_from_config('priotize_masks'))
        while True:
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])

            if containsMask:
                if 1 in np.unique(label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]):
                    break
            else: 
                break
        
        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]

        return img, label

    def __getitem__(self, idx):
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        znormalisation = torchio.ZNormalization()

        img = self.data.get_data(key=self.images_list[idx])
        if not img:
            img_name, p_id, img_id = self.images_list[idx]

            label_name, _, _ = self.labels_list[idx]

            img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, img_name))
            # 2D
            if self.slice is not None:
                if len(img.shape) == 4:
                    img = img[..., 0]
                if self.exp.get_from_config('rescale') is not None and self.exp.get_from_config('rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.slice == 0:
                    img, label = img[img_id, :, :], label[img_id, :, :]
                elif self.slice == 1:
                    img, label = img[:, img_id, :], label[:, img_id, :]
                elif self.slice == 2:
                    img, label = img[:, :, img_id], label[:, :, img_id]
                # Remove 4th dimension if multiphase
                if len(img.shape) == 4:
                    img = img[...,0] 
                img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
            # 3D
            else:
                if len(img.shape) == 4:
                    img = img[..., 0]
                img = np.expand_dims(img, axis=0)
                img = rescale(img) 
                img = np.squeeze(img)
                if self.exp.get_from_config('rescale') is not None and self.exp.get_from_config('rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.exp.get_from_config('keep_original_scale') is not None and self.exp.get_from_config('keep_original_scale'):
                    img, label = self.preprocessing3d(img), self.preprocessing3d(label, isLabel=True)  
                # Add dim to label
                if len(label.shape) == 3:
                    label = np.expand_dims(label, axis=-1)
            img_id = "_" + str(p_id) + "_" + str(img_id)
            
            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])
           

        id, img, label = img

        size = self.size 
        
        # Create patches from full resolution
        if self.exp.get_from_config('patchify') is not None and self.exp.get_from_config('patchify') is True and self.state == "train": 
            img, label = self.patchify(img, label) 

        if len(size) > 2:
            size = size[0:2] 

        # Normalize image
        img = np.expand_dims(img, axis=0)
        if np.sum(img) > 0:
            img = znormalisation(img)
        img = rescale(img) 
        img = img[0]

        # Merge labels -> For now single label
        label[label > 0] = 1

        # Number of defined channels
        if len(self.size) == 2:
            img = img[..., :self.exp.get_from_config('input_channels')]
            label = label[..., :self.exp.get_from_config('output_channels')]

        return (id, img, label)


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
            Reuses cubic interpolation for images and nearest for labels.
        """
        size = (self.size[0], self.size[1])
        size2 = (self.size[2], self.size[0])
        interp = cv2.INTER_NEAREST if isLabel else cv2.INTER_CUBIC

        resized = np.zeros((self.size[0], self.size[1], img.shape[2]), dtype=np.float32)
        for z in range(img.shape[2]):
            resized[:, :, z] = cv2.resize(img[:, :, z], dsize=size, interpolation=interp)

        if len(self.size) == 3:
            tmp = resized
            resized = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=np.float32)
            for y in range(tmp.shape[1]):
                resized[:, y, :] = cv2.resize(tmp[:, y, :], dsize=size2, interpolation=interp)
        return resized

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
        for _ in range(50):  # bounded retries to find a region-containing patch
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])
            if not contains_mask:
                break
            patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], region]
            if patch.max() > 0:
                break
            # Fall back to WT if the target region isn't found (ET can be tiny/absent)
            wt_patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0]
            if region != 0 and wt_patch.max() > 0:
                # keep looking for the target region, but remember this WT-valid pos
                continue

        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        return img, label
