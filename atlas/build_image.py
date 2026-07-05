#!/usr/bin/env python3
"""Download remaining layers & docker load Atlas image"""
import json, os, urllib.request, hashlib, tarfile, shutil, subprocess

IMAGE = "avarok/atlas-gb10"
TAG = "latest"
WORKDIR = "/home/nvidia/vLLM/atlas/layers"

os.makedirs(WORKDIR, exist_ok=True)

# Copy existing layers from /tmp
for f in os.listdir("/tmp/atlas-load-cqjg0tn7/"):
    if f.endswith(".tar.gz"):
        shutil.copy2(os.path.join("/tmp/atlas-load-cqjg0tn7/", f), os.path.join(WORKDIR, f))

# Copy large layer
large = os.path.join(WORKDIR, "sha256_eeb7c758692890da3653ab7f03e22acd54ee9ee319df9c3c25695b999a7d156b.tar.gz")
if not os.path.exists(large) and os.path.exists("/tmp/atlas-large-layer.tar.gz"):
    shutil.copy2("/tmp/atlas-large-layer.tar.gz", large)

# Get token & manifest
req = urllib.request.Request(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{IMAGE}:pull")
with urllib.request.urlopen(req, timeout=30) as f:
    token = json.loads(f.read())["token"]

headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.docker.distribution.manifest.v2+json"}
req = urllib.request.Request(f"https://registry-1.docker.io/v2/{IMAGE}/manifests/{TAG}", headers=headers)
with urllib.request.urlopen(req, timeout=30) as f:
    manifest = json.loads(f.read())

# Get config
cd = manifest["config"]["digest"]
headers["Accept"] = "application/vnd.docker.container.image.v1+json"
req = urllib.request.Request(f"https://registry-1.docker.io/v2/{IMAGE}/blobs/{cd}", headers=headers)
with urllib.request.urlopen(req, timeout=60) as f:
    config = json.loads(f.read())

# Existing layers
exist = {f.replace("sha256_", "sha256:").replace(".tar.gz", "") for f in os.listdir(WORKDIR) if f.endswith(".tar.gz")}

# Download missing
for i, layer in enumerate(manifest["layers"]):
    d = layer["digest"]
    if d in exist: continue
    fn = d.replace(":", "_") + ".tar.gz"
    fp = os.path.join(WORKDIR, fn)
    print(f"DL L{i+1} ({layer['size']/1024/1024:.0f}MB)...", end="", flush=True)
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{IMAGE}/blobs/{d}",
                                  headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=600) as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    assert h == d.split(":")[1], f"hash mismatch {h[:16]} vs {d.split(':')[1][:16]}"
    open(fp, "wb").write(data)
    print(" ok")

# Write config
cf = cd.replace(":", "_") + ".json"
with open(os.path.join(WORKDIR, cf), "w") as f:
    json.dump(config, f)

# Layer list
lf = [l["digest"].replace(":", "_") + ".tar.gz" for l in manifest["layers"]]

# Manifest
with open(os.path.join(WORKDIR, "manifest.json"), "w") as f:
    json.dump([{"Config": cf, "Layers": lf, "RepoTags": [f"{IMAGE}:{TAG}"]}], f)

# Create tar
tp = "/home/nvidia/vLLM/atlas/atlas-gb10.tar"
with tarfile.open(tp, "w") as tar:
    for fn in [cf] + lf + ["manifest.json"]:
        p = os.path.join(WORKDIR, fn)
        if os.path.exists(p): tar.add(p, arcname=fn)
        else: print(f"!! MISSING {fn}")

print(f"\nTar: {os.path.getsize(tp)/1024/1024:.0f}MB")

# docker load
r = subprocess.run(["docker", "load", "-i", tp], capture_output=True, text=True, timeout=600)
print(r.stdout.strip())
if r.stderr: print(f"ERR: {r.stderr[:200]}")
print("DONE - Atlas loaded!")
