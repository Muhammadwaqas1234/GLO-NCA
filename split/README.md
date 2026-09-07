# Master patient split

`master_split.json` is the ONE canonical patient-level train/validation/test
split shared by every experiment (A0, A1, A2, A3, Final). It holds **patient IDs
only** — never image data — so it is safe and useful to commit for
reproducibility.

It is **not** present until you create it on the machine that has the real
BraTS dataset:

```bash
python scripts/create_master_split.py --data-root "$VM_DATA_DIR"
python scripts/check_split.py --split split/master_split.json --data-root "$VM_DATA_DIR"
```

All five configs reference `split/master_split.json` via `data.split_file`. If
the file is missing, every training run **fails loudly** rather than silently
generating a fresh split — this prevents experimental drift.

`.gitignore` allows `master_split.json` to be committed (the real, generated
one) while ignoring any other files placed here. Commit the real split once it
is generated so the exact split used for the thesis is archived in git.
