"""Run unittest classes and zero-argument test functions through one entry point."""

import argparse
import inspect
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectLoader(unittest.TestLoader):
    def loadTestsFromModule(self, module, *, pattern=None):
        suite = super().loadTestsFromModule(module, pattern=pattern)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__:
                if inspect.signature(function).parameters:
                    raise TypeError(f"{module.__name__}.{name} requires unsupported fixtures")
                suite.addTest(unittest.FunctionTestCase(function))
        return suite


def collect(scope="all", pattern="test*.py"):
    suite = unittest.TestSuite()
    groups = ("MyFlows/tests", "tests") if scope == "all" else (
        "MyFlows/tests" if scope == "framework" else "tests",
    )
    for directory in groups:
        loader = ProjectLoader()
        # Separate loaders avoid unittest reusing the first discovery root.
        suite.addTests(loader.discover(str(ROOT / directory), pattern=pattern))
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "framework", "apps"), default="all")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for iteration in range(args.repeat):
        suite = collect(args.scope, args.pattern)
        print(f"Run {iteration + 1}/{args.repeat}: {suite.countTestCases()} collected", flush=True)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
