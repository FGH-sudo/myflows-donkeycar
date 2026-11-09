import unittest

from benchmark.memory_study import CONFIGS
from benchmark.perf_gate import violations
from tools.sanitize_native import parse_result


class StageThreeGateTest(unittest.TestCase):
    def test_strict_shape_and_step_limits_and_missing_measurements(self):
        ratios = {name: 1.1 for name, *_ in CONFIGS}
        rows = [dict(model=model,part=part,layout=layout,m=4,n=3,k=2,register_ms=1.2,cublas_ms=1.)
                for model in ("w16","w64") for part,layout in (("forward","NT"),("dW","TN"),("dX","NN"))]
        self.assertEqual(violations(ratios,rows),[])
        rows[0]["register_ms"]=1.21
        ratios[CONFIGS[0][0]]=1.11
        self.assertEqual(len(violations(ratios,rows)),2)
        with self.assertRaises(ValueError):
            violations(ratios,rows[1:])
        rows[0]["register_ms"]=float("nan")
        with self.assertRaises(ValueError):
            violations(ratios,rows)

    def test_sanitizer_requires_tests_and_an_explicit_clean_tool_summary(self):
        log="Ran 22 tests in 1.5s\n\nOK\n"
        self.assertTrue(parse_result("memcheck",log,"ERROR SUMMARY: 0 errors",0)["passed"])
        for text in ("", "RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)"):
            self.assertFalse(parse_result("racecheck",log,text,0)["passed"])
        self.assertFalse(parse_result("synccheck","OK (skipped=22)","ERROR SUMMARY: 0 errors",0)["passed"])
