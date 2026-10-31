"""Download original SomBench ice-prospectivity GeoTIFFs at the pinned dataset revision.

Public, ungated dataset: no credential is used (``token=False``; any cached login is ignored).
Every file is verified against the sha256 that Hugging Face publishes for it (LFS metadata) and
recorded, with size and revision, in ``<dest>/download_manifest.json``. Only the files named on
the command line are fetched; no model checkpoint is ever downloaded here.

    python scripts/prepare_sombench.py --patches patch_0001_0002_S_80S
    python scripts/prepare_sombench.py --patches-from-split test --pole south --layers inputs
    # evaluator-only expert target: keep it OUTSIDE the observation directory
    python scripts/prepare_sombench.py --patches patch_0001_0002_S_80S --layers PRO \\
        --dest data/evaluation_inputs/prospectivity_target
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = "nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression"
REVISION = "78040503ceaa68107829f48114774112e8ee940d"
INPUT_LAYERS = ["ASP_SIN_COS", "CUR", "DICE", "LPSR", "LPSR_DEN", "LPSR_DIS", "SLOPE", "TMAX"]
SPLIT_FILES = [
    "train_filtered.txt",
    "val_filtered.txt",
    "test_filtered.txt",
    "band_statistics.json",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path("data/sombench/prospectivity_dataset"))
    ap.add_argument("--patches", nargs="*", default=[])
    ap.add_argument("--patches-from-split", choices=["test"], default=None)
    ap.add_argument("--pole", choices=["south", "north"], default="south")
    ap.add_argument(
        "--layers", default="inputs", help="'inputs' (the 8 model inputs) or comma-separated names"
    )
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    layers = INPUT_LAYERS if args.layers == "inputs" else args.layers.split(",")
    args.dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        print(f"revision mismatch: {info.sha} != {REVISION}", file=sys.stderr)
        return 1
    meta = {s.rfilename: s for s in info.siblings}

    def fetch(name: str) -> Path:
        if name not in meta:
            raise SystemExit(f"{name} is not in dataset revision {REVISION}")
        path = Path(
            hf_hub_download(
                REPO, name, repo_type="dataset", revision=REVISION, local_dir=args.dest, token=False
            )
        )
        got = sha256_file(path)
        lfs = getattr(meta[name], "lfs", None)
        expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        method = "lfs-sha256"
        if not expected:
            # small files are plain git blobs: verify the git blob id (sha1 of "blob <n>\0"+data)
            blob_id = getattr(meta[name], "blob_id", None)
            data = path.read_bytes()
            git_sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            if blob_id and git_sha1 != blob_id:
                path.unlink()
                raise SystemExit(f"{name}: git blob {git_sha1} != published {blob_id}; removed")
            method = "git-blob-sha1" if blob_id else "unverified"
        elif got != expected:
            path.unlink()
            raise SystemExit(f"{name}: sha256 {got} != published {expected}; file removed")
        records[name] = {
            "size": path.stat().st_size,
            "sha256": got,
            "published_lfs_sha256": expected,
            "hub_verification": method,
        }
        return path

    records: dict[str, dict] = {}
    for n in SPLIT_FILES:
        fetch(n)
    patches = list(args.patches)
    if args.patches_from_split:
        suffix = "_S_80S" if args.pole == "south" else "_80N"
        ids = (args.dest / "test_filtered.txt").read_text().split()
        patches += [p for p in ids if p.endswith(suffix)]
    declared = set()
    for split_file in ("train_filtered.txt", "val_filtered.txt", "test_filtered.txt"):
        declared |= set((args.dest / split_file).read_text().split())
    for p in dict.fromkeys(patches):
        if p not in declared:
            raise SystemExit(f"{p} is not listed in any split file")
        for layer in layers:
            fetch(f"{p}_{layer}.tif")

    manifest_path = args.dest / "download_manifest.json"
    old = json.loads(manifest_path.read_text())["files"] if manifest_path.exists() else {}
    old.update(records)
    manifest_path.write_text(
        json.dumps(
            {"repo": REPO, "revision": REVISION, "license": "CC-BY-4.0", "files": old},
            indent=2,
            sort_keys=True,
        )
    )
    total = sum(r["size"] for r in records.values())
    print(f"fetched/verified {len(records)} files, {total} bytes this run -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
