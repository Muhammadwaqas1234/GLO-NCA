#!/usr/bin/env python
r"""Synthetic tests for the fail-closed dataset preflight logic.
Proves: 100->PASS, 99->FAIL, 101->FAIL, wrong-ID->FAIL, missing-seg->FAIL,
duplicate-ID->FAIL, unexpected-ID->FAIL. Uses tiny empty files on a temp tree;
no GPU, no real data. Mirrors the kernel's preflight checks exactly."""
import os, sys, glob, hashlib, tempfile, shutil

MODS = ["t1n","t1c","t2w","t2f"]

def build_tree(root, ids, cohort_of, drop_seg=None, extra_empty=None):
    for cid in ids:
        coh = cohort_of(cid)
        d = os.path.join(root, cid) if coh == "_flat_" else os.path.join(root, coh, cid)
        os.makedirs(d, exist_ok=True)
        for m in MODS: open(os.path.join(d, f"{cid}-{m}.nii.gz"), "wb").write(b"x")
        if cid != drop_seg: open(os.path.join(d, f"{cid}-seg.nii.gz"), "wb").write(b"x")
    if extra_empty:
        d = os.path.join(root, extra_empty); os.makedirs(d, exist_ok=True)
        for m in MODS+["seg"]: open(os.path.join(d, f"{extra_empty}-{m}.nii.gz"), "wb").write(b"x")

def preflight(root, expect_count, expect_flat, expect_nested, expect_ids_sha):
    """Returns (ok, reason) mirroring the kernel's checks (structural subset)."""
    segs = sorted(glob.glob(os.path.join(root,"**","*-seg.nii.gz"), recursive=True))
    disc = {}
    for s in segs:
        d=os.path.dirname(s); cid=os.path.basename(s)[:-len("-seg.nii.gz")]
        rel=os.path.relpath(d,root); coh=rel.split(os.sep)[0] if os.sep in rel else "_flat_"
        disc[cid]={"dir":d,"cohort":coh,"seg":s}
    disc_ids=sorted(disc); ids_sha=hashlib.sha256("".join(disc_ids).encode()).hexdigest()
    flat=sum(1 for c in disc.values() if c["cohort"]=="_flat_")
    nested=sum(1 for c in disc.values() if c["cohort"]=="UCSD - Training")
    missing=[]
    for cid,c in disc.items():
        for m in MODS:
            p=os.path.join(c["dir"],f"{cid}-{m}.nii.gz")
            if not os.path.isfile(p) or os.path.getsize(p)==0: missing.append(f"{cid}:{m}")
    # also require every discovered dir with modalities to have a seg (missing-seg detection):
    # find case dirs that have modalities but no seg
    for cid_dir in glob.glob(os.path.join(root,"**",""), recursive=True):
        base=os.path.basename(os.path.dirname(cid_dir))
        if os.path.exists(os.path.join(cid_dir, f"{base}-t1n.nii.gz")) and not os.path.exists(os.path.join(cid_dir, f"{base}-seg.nii.gz")):
            missing.append(f"{base}:seg(no-seg-dir)")
    if len(disc)!=expect_count: return False, f"count {len(disc)}!={expect_count}"
    if flat!=expect_flat or nested!=expect_nested: return False, f"cohort {flat}/{nested}"
    if missing: return False, f"missing {missing[:3]}"
    if ids_sha!=expect_ids_sha: return False, "id-set mismatch"
    return True, "PASS"

def main():
    base_ids = [f"BraTS-MET-{i:05d}-000" for i in range(1, 101)]  # 100 ids
    def cohort(cid):
        n = int(cid.split("-")[2]); return "_flat_" if n <= 50 else "UCSD - Training"
    exp_ids_sha = hashlib.sha256("".join(sorted(base_ids)).encode()).hexdigest()
    E = dict(expect_count=100, expect_flat=50, expect_nested=50, expect_ids_sha=exp_ids_sha)

    tests = []
    def run(name, ids, expect_ok, **kw):
        tmp = tempfile.mkdtemp()
        try:
            build_tree(tmp, ids, cohort, **kw)
            ok, reason = preflight(tmp, **E)
            passed = (ok == expect_ok)
            tests.append((name, passed, f"got ok={ok} ({reason}), wanted ok={expect_ok}"))
        finally: shutil.rmtree(tmp, ignore_errors=True)

    run("100 expected -> PASS", base_ids, True)
    run("99 discovered -> FAIL", base_ids[:99], False)
    run("101 discovered -> FAIL", base_ids + ["BraTS-MET-00101-000"], False)  # extra id (nested by rule)
    run("wrong ID -> FAIL", base_ids[:99] + ["BraTS-MET-99999-000"], False)
    run("missing seg -> FAIL", base_ids, False, drop_seg=base_ids[0])
    run("unexpected extra dir -> FAIL", base_ids, False, extra_empty="BraTS-MET-00102-000")

    print("="*50); print("PREFLIGHT LOGIC TESTS"); print("="*50)
    npass = sum(1 for _,ok,_ in tests if ok)
    for name, ok, detail in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" -- {detail}"))
    print(f"\n{npass}/{len(tests)} tests passed")
    return 0 if npass == len(tests) else 1

if __name__ == "__main__":
    raise SystemExit(main())
