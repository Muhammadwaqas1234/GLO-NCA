"""Data pipeline: dataset, data-root discovery and the random split."""
from __future__ import annotations

import os
import random

import cv2
import numpy as np

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS


class BraTS_FG(Dataset_NiiGz_3D_BraTS):
    """BraTS loader with foreground crop, cubic resize and nonzero z-norm."""

    use_foreground_crop = True
    use_nonzero_norm = True

    @staticmethod
    def _foreground_bbox(vol_stack):
        fg = np.any(vol_stack > 0, axis=-1)
        if not fg.any():
            return None
        xs = np.where(fg.any(axis=(1, 2)))[0]
        ys = np.where(fg.any(axis=(0, 2)))[0]
        zs = np.where(fg.any(axis=(0, 1)))[0]
        return xs[0], xs[-1] + 1, ys[0], ys[-1] + 1, zs[0], zs[-1] + 1

    def _resize_to(self, vol, size, is_label=False):
        interp = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
        out = np.zeros((size[0], size[1], vol.shape[2]), np.float32)
        for z in range(vol.shape[2]):
            out[:, :, z] = cv2.resize(vol[:, :, z], dsize=(size[1], size[0]), interpolation=interp)
        tmp, out = out, np.zeros(size, np.float32)
        for y in range(tmp.shape[1]):
            out[:, y, :] = cv2.resize(tmp[:, y, :], dsize=(size[2], size[0]), interpolation=interp)
        return out

    def __getitem__(self, idx):
        key = self.images_list[idx]
        cached = self.data.get_data(key=key)
        if not cached:
            folder_name, p_id, _ = key
            folder = os.path.join(self.images_path, folder_name)
            raw = np.stack([self.load_item(self._find_modality_file(folder, folder_name, m))
                            for m in self.MODALITIES], axis=-1)
            seg = self.load_item(self._find_modality_file(folder, folder_name, self.SEG_SUFFIX))
            if self.use_foreground_crop:
                bbox = self._foreground_bbox(raw)
                if bbox is not None:
                    x0, x1, y0, y1, z0, z1 = bbox
                    raw = raw[x0:x1, y0:y1, z0:z1, :]; seg = seg[x0:x1, y0:y1, z0:z1]
            size = tuple(self.size)
            img = np.stack([self._resize_to(raw[..., c], size) for c in range(raw.shape[-1])], axis=-1)
            seg = self._resize_to(seg, size, is_label=True)
            label = self._labels_to_regions(seg)
            self.data.set_data(key=key, data=("_" + str(p_id) + "_0", img, label))
            cached = self.data.get_data(key=key)

        img_id, img, label = cached
        if self.exp.get_from_config('patchify') is True and self.state == "train":
            img, label = self.patchify_multimodal(img, label)
        # No augmentation is applied here.
        if self.use_nonzero_norm:
            out = np.empty_like(img, dtype=np.float32)
            for c in range(img.shape[-1]):
                ch = img[..., c]; mask = ch > 0
                out[..., c] = np.where(mask, (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8), 0.0) \
                    if mask.sum() > 0 else ch
            img = out
        return (img_id, img.astype(np.float32), label.astype(np.float32))


def find_data_root(base="/kaggle/input"):
    """First folder under ``base`` that holds at least two case folders with scans."""
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
    """Seeded random 70/15/15 split of the case folders in ``data_root``."""
    # Case folders only: a folder that holds scans (skips nested cohort folders).
    pats = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
                  and any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(os.path.join(data_root, d))))
    random.Random(seed).shuffle(pats)
    if n_patients:
        pats = pats[:n_patients]
    n = len(pats); a, b = int(n*0.70), int(n*0.15)
    return pats[:a], pats[a:a+b], pats[a+b:]
