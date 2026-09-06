"""Archive three independent full regression processes with their source snapshot."""

import argparse
import re
import subprocess
import sys

from benchmark.stage1_common import ROOT, RunArtifacts
from MyFlows.core.device import cuda_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scope", choices=("all", "framework"), default="all")
    args = parser.parse_args()
    with RunArtifacts(args.out_dir, vars(args)) as run:
        if not cuda_available():
            raise RuntimeError("the stage-one regression gate requires an actual CUDA device")
        run.results["regressions"] = []
        for repeat in range(3):
            command = [sys.executable, "-X", "utf8", "-m", "tools.run_tests", "--scope", args.scope]
            result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", timeout=180)
            (run.path / f"tests-{repeat + 1}.log").write_text(result.stdout, encoding="utf-8")
            counts = re.findall(r"Ran (\d+) tests", result.stdout)
            run.results["regressions"].append({"repeat": repeat, "exitcode": result.returncode,
                                               "count": int(counts[-1]) if counts else None,
                                               "skips": re.findall(r"^.*\.\.\. skipped .*$", result.stdout, flags=re.MULTILINE)})
            if result.returncode or not counts:
                raise RuntimeError(f"regression process {repeat + 1} failed; see its test log")
            print(run.results["regressions"][-1], flush=True)
        run.notes.append("Each repetition starts a fresh Python process and runs both unittest classes and function tests. GPU availability is mandatory for this gate.")


if __name__ == "__main__":
    main()
