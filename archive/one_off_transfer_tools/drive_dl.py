#!/usr/bin/env python3
"""Auth-free download of a PUBLIC large Google Drive file via the virus-scan
confirm-token flow. Streams to disk, verifies ZIP magic. Resumable-safe (writes
to .part then renames). No credentials, no external tools."""
import sys, re, os, urllib.request, http.cookiejar, urllib.parse
FID = "1gkz4kM89PUdqI4rbpi1KN5TYGfBGYe5k"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/batch1.zip"
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", "Mozilla/5.0")]
r = op.open("https://drive.google.com/uc?export=download&id=" + FID, timeout=60)
html = r.read().decode("utf-8", "ignore")
form = re.search(r'action="(https://drive\.usercontent\.google\.com/download[^"]*)"', html)
if not form:
    # already a direct binary?
    print("NO_CONFIRM_FORM (may be small/direct)"); sys.exit(3)
action = form.group(1).replace("&amp;", "&")
params = {m.group(1): m.group(2) for m in re.finditer(r'name="([^"]+)" value="([^"]*)"', html)}
dl = action + ("&" if "?" in action else "?") + urllib.parse.urlencode(params)
req = urllib.request.Request(dl, headers={"User-Agent": "Mozilla/5.0"})
resp = op.open(req, timeout=120)
total = int(resp.headers.get("content-length") or 0)
ct = resp.headers.get("content-type", "")
print(f"content-type={ct} total_bytes={total}")
if "text/html" in ct:
    print("GOT_HTML_NOT_FILE"); sys.exit(4)
tmp = OUT + ".part"
first = resp.read(4)
if first[:2] != b"PK":
    print("NOT_ZIP first4=%r" % first); sys.exit(5)
done = 4
with open(tmp, "wb") as f:
    f.write(first)
    while True:
        chunk = resp.read(8 * 1024 * 1024)
        if not chunk:
            break
        f.write(chunk); done += len(chunk)
        if done % (1024 * 1024 * 1024) < 8 * 1024 * 1024:
            print(f"  downloaded {done/1e9:.1f} GB", flush=True)
os.replace(tmp, OUT)
sz = os.path.getsize(OUT)
print(f"DOWNLOAD_DONE bytes={sz} expected={total} match={sz==total or total==0}")
