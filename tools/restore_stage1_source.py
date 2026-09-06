"""Restore a recorded source snapshot into a new directory without Git history."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def restore(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    pending = {}
    with zipfile.ZipFile(run_dir / "source-at-run.zip") as archive:
        for repo, state in manifest["source"].items():
            if repo not in ("root", "MyFlows"):
                raise ValueError(f"unknown repository {repo}")
            base = out_dir if repo == "root" else out_dir / "MyFlows"
            for relative, expected in state["files_sha256"].items():
                destination = (base / relative).resolve()
                if not destination.is_relative_to(base):
                    raise ValueError("source path escapes destination")
                contents = archive.read(f"{repo}/{relative}")
                if hashlib.sha256(contents).hexdigest() != expected:
                    raise ValueError(f"source checksum mismatch: {repo}/{relative}")
                pending[destination] = contents
    out_dir.mkdir(parents=True, exist_ok=False)
    for path, contents in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    (out_dir / ".stage1-source-origin.json").write_text(
        json.dumps(manifest["source"], indent=2, ensure_ascii=False), encoding="utf-8")
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    count = restore(args.run_dir, args.out_dir)
    print(f"Restored and verified {count} files to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
