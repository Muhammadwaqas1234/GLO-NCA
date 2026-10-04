r"""Data pipeline: the BraTS dataset, data-root discovery and the random split."""
from __future__ import annotations

import hashlib
import os
import random

import cv2
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

CACHE_MODES = ("memory", "disk", "none")


class _ConfigView:
    r"""Picklable stand-in for the experiment inside DataLoader worker processes."""
    def __init__(self, config):
        r"""Keep a copy of the config dictionary."""
        self.config = dict(config)

    def get_from_config(self, tag):
        r"""Get a value from the config, or None when absent."""
        return self.config.get(tag)


class Dataset_BraTS_Foreground(Dataset):
    r"""BraTS loader: foreground crop, cubic resize, nonzero z-norm, patches and augmentation."""

    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
    SEG_SUFFIX = "seg"
    use_foreground_crop = True
    use_nonzero_norm = True
    augment = False

    def __init__(self, cache="memory", cache_dir=None):
        r"""Choose where preprocessed volumes are kept: memory, disk (cache_dir) or none."""
        if cache not in CACHE_MODES:
            raise ValueError(f"cache must be one of {CACHE_MODES}, got {cache!r}")
        if cache == "disk" and not cache_dir:
            raise ValueError("cache 'disk' needs a cache_dir")
        self.cache, self.cache_dir = cache, cache_dir
        self._memory = {}
        self.images_list, self.length, self.state = [], 0, "train"

    # Experiment interface.
    def set_size(self, size):
        r"""Working size (X, Y, Z) of the resampled volumes."""
        self.size = tuple(size)

    def set_experiment(self, experiment):
        r"""Attach the experiment; the case folders live under its img_path."""
        self.exp = experiment
        self.images_path = experiment.get_from_config('img_path')

    def set_paths(self, images_list):
        r"""Case entries of the active split."""
        self.images_list = images_list
        self.length = len(images_list)

    def set_state(self, state):
        r"""Switch between the train, val and test splits."""
        self.state = state

    def __len__(self):
        r"""Number of cases in the active split."""
        return self.length

    def __getstate__(self):
        r"""Pickle without the experiment (models, optimizers), for spawned worker processes."""
        state = self.__dict__.copy()
        if "exp" in state:
            state["exp"] = _ConfigView(self.exp.config)
        return state

    # Loading and preprocessing.
    def _find_modality_file(self, folder, patient, suffix):
        r"""Locate a modality/seg file in a patient folder, tolerant to naming."""
        # Common explicit names (underscore or hyphen separator, .nii.gz/.nii).
        for sep in ("_", "-"):
            for ext in (".nii.gz", ".nii"):
                candidate = os.path.join(folder, f"{patient}{sep}{suffix}{ext}")
                if os.path.exists(candidate):
                    return candidate
        # Fallback: any file ending with that modality suffix, either separator.
        endings = (f"_{suffix}.nii.gz", f"_{suffix}.nii", f"-{suffix}.nii.gz", f"-{suffix}.nii")
        for f in os.listdir(folder):
            if f.lower().endswith(endings):
                return os.path.join(folder, f)
        raise FileNotFoundError(f"Could not find '{suffix}' volume for patient '{patient}' in {folder}")

    def load_item(self, path):
        r"""Load a single nii/nii.gz volume as a float numpy array."""
        return nib.load(path).get_fdata()

    @staticmethod
    def _foreground_bbox(vol_stack):
        r"""Bounding box of the non-zero brain across all modalities."""
        fg = np.any(vol_stack > 0, axis=-1)
        if not fg.any():
            return None
        xs = np.where(fg.any(axis=(1, 2)))[0]
        ys = np.where(fg.any(axis=(0, 2)))[0]
        zs = np.where(fg.any(axis=(0, 1)))[0]
        return xs[0], xs[-1] + 1, ys[0], ys[-1] + 1, zs[0], zs[-1] + 1

    @staticmethod
    def _labels_to_regions(seg):
        r"""Convert raw BraTS labels into the nested WT/TC/ET regions."""
        # ET is label 4 (3 in some remapped copies).
        et = np.logical_or(seg == 4, seg == 3)
        ncr = (seg == 1)
        ed = (seg == 2)
        wt = np.logical_or(np.logical_or(ncr, ed), et)   # whole tumour
        tc = np.logical_or(ncr, et)                       # tumour core
        return np.stack([wt, tc, et], axis=-1).astype(np.float32)

    def _resize_to(self, vol, size, is_label=False):
        r"""Resize a volume slice-wise: cubic for images, nearest for labels."""
        interp = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
        out = np.zeros((size[0], size[1], vol.shape[2]), np.float32)
        for z in range(vol.shape[2]):
            out[:, :, z] = cv2.resize(vol[:, :, z], dsize=(size[1], size[0]), interpolation=interp)
        tmp, out = out, np.zeros(size, np.float32)
        for y in range(tmp.shape[1]):
            out[:, y, :] = cv2.resize(tmp[:, y, :], dsize=(size[2], size[0]), interpolation=interp)
        return out

    def _preprocess(self, folder_name):
        r"""Load, crop and resample one case to (image, regions) at the working size."""
        folder = os.path.join(self.images_path, folder_name)
        raw = np.stack([self.load_item(self._find_modality_file(folder, folder_name, m))
                        for m in self.MODALITIES], axis=-1)
        seg = self.load_item(self._find_modality_file(folder, folder_name, self.SEG_SUFFIX))
        if self.use_foreground_crop:
            bbox = self._foreground_bbox(raw)
            if bbox is not None:
                x0, x1, y0, y1, z0, z1 = bbox
                raw = raw[x0:x1, y0:y1, z0:z1, :]
                seg = seg[x0:x1, y0:y1, z0:z1]
        size = tuple(self.size)
        img = np.stack([self._resize_to(raw[..., c], size) for c in range(raw.shape[-1])], axis=-1)
        label = self._labels_to_regions(self._resize_to(seg, size, is_label=True))
        return img, label

    def _cache_path(self, folder_name):
        r"""Disk-cache file for one case; the name encodes every setting that changes the arrays."""
        tag = f"{self.size}|{self.use_foreground_crop}|{self.MODALITIES}"
        digest = hashlib.sha1(tag.encode()).hexdigest()[:10]
        return os.path.join(self.cache_dir, f"{folder_name}_{digest}.npz")

    def _load_case(self, key):
        r"""Preprocessed (image, regions) for one case, through the configured cache."""
        folder_name = key[0]
        if self.cache == "memory":
            if key not in self._memory:
                self._memory[key] = self._preprocess(folder_name)
            return self._memory[key]
        if self.cache == "disk":
            path = self._cache_path(folder_name)
            if os.path.exists(path):
                with np.load(path) as z:
                    return z["img"], z["label"]
            img, label = self._preprocess(folder_name)
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp.npz"
            np.savez(tmp, img=img, label=label)
            os.replace(tmp, path)
            return img, label
        return self._preprocess(folder_name)

    def __getitem__(self, idx):
        r"""One case: cached volume, then patch, augmentation and nonzero z-norm in training."""
        key = self.images_list[idx]
        img, label = self._load_case(key)
        img_id = "_" + str(key[1]) + "_0"
        if self.exp.get_from_config('patchify') is True and self.state == "train":
            img, label = self.patchify_multimodal(img, label)
        augment = self.augment and self.state == "train"
        if augment:
            img, label = self._augment_spatial(img, label)
        if self.use_nonzero_norm:
            img = self.nonzero_norm(img)
        if augment:
            img = self._augment_intensity(img)
        return (img_id, img.astype(np.float32), label.astype(np.float32))

    @staticmethod
    def nonzero_norm(img):
        r"""Per-channel z-normalisation over brain (non-zero) voxels; background stays zero."""
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[-1]):
            ch = img[..., c]
            mask = ch > 0
            out[..., c] = np.where(mask, (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8), 0.0) \
                if mask.sum() > 0 else ch
        return out

    def patchify_multimodal(self, img, label):
        r"""Random 3D patch of the working size, biased toward a tumour region."""
        size = self.size
        prioritize = self.exp.get_from_config('priotize_masks')
        contains_mask = prioritize is not None and (random.uniform(0, 1) < prioritize)
        # Region to bias the patch toward: 0=WT (default), 1=TC, 2=ET.
        region = self.exp.get_from_config('prioritize_region')
        region = 0 if region is None else int(region)

        pos_x = pos_y = pos_z = 0
        for _ in range(50):  # bounded retries for a patch that contains the region
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])
            if not contains_mask:
                break
            patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], region]
            if patch.max() > 0:
                break

        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
        return img, label

    @staticmethod
    def _augment_spatial(img, label):
        r"""Random axis flips and in-plane 90-degree rotations, applied to image and label alike."""
        for axis in range(3):
            if random.random() < 0.5:
                img, label = np.flip(img, axis), np.flip(label, axis)
        if img.shape[0] == img.shape[1]:
            k = random.randint(0, 3)
            img, label = np.rot90(img, k, axes=(0, 1)), np.rot90(label, k, axes=(0, 1))
        return np.ascontiguousarray(img), np.ascontiguousarray(label)

    @staticmethod
    def _augment_intensity(img, scale=0.1, shift=0.1):
        r"""Per-channel random intensity scale and shift on brain voxels only."""
        out = img.astype(np.float32, copy=True)
        for c in range(out.shape[-1]):
            mask = out[..., c] != 0
            out[..., c][mask] = (out[..., c][mask] * random.uniform(1 - scale, 1 + scale)
                                 + random.uniform(-shift, shift))
        return out


def find_data_root(base="/kaggle/input"):
    r"""First folder under ``base`` that holds at least two case folders with scans."""
    for root, dirs, files in os.walk(base):
        c = 0
        for d in dirs:
            try:
                if any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(os.path.join(root, d))):
                    c += 1
            except Exception:
                pass
        if c >= 2:
            return root, c
    return None, 0


def make_split(data_root, seed, n_patients=None):
    r"""Seeded random 70/15/15 split of the case folders in ``data_root``."""
    # Case folders only: a folder that holds scans (skips nested cohort folders).
    pats = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
                  and any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(os.path.join(data_root, d))))
    random.Random(seed).shuffle(pats)
    if n_patients:
        pats = pats[:n_patients]
    n = len(pats); a, b = int(n*0.70), int(n*0.15)
    return pats[:a], pats[a:a+b], pats[a+b:]
