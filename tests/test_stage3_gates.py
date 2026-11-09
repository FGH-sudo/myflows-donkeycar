import unittest

from benchmark.gemm_ncu import parse_csv, summarize
from benchmark.memory_study import CONFIGS
from benchmark.perf_gate import gemm_ratio, violations
from tools.sanitize_native import parse_result

NCU_HEADER = ('"ID","Process ID","thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",'
              '"Kernel Name","Block Size","gpu__time_duration.sum","sm__cycles_elapsed.max"\n'
              '"","","","","","ns","cycle"\n')


def ncu_row(row_id, name, ns, cycles):
    return f'"{row_id}","7","28112  ""<default domain>:x"" ","{name}","(256, 1, 1)","{ns:,}","{cycles:,}"\n'


class NsightGemmTest(unittest.TestCase):
    def test_kernels_of_one_range_are_one_call_and_repeats_are_kept_in_order(self):
        text = "==PROF== Connected\n" + NCU_HEADER + "".join([
            ncu_row(0, "g0:register:0/gemm_register_p8_tn_v1", 18000, 34000),
            ncu_row(1, "g0:register:0/gemm_split_reduce", 3000, 5600),
            ncu_row(2, "g0:cublas:0/ampere_sgemm_64x64_tn", 16000, 30000),
            ncu_row(3, "g0:cublas:1/ampere_sgemm_64x64_tn", 16500, 31000),
            ncu_row(4, "g0:register:1/gemm_register_p8_tn_v1", 18100, 34100),
            ncu_row(5, "g0:register:1/gemm_split_reduce", 2900, 5500),
            ncu_row(6, "g0:register:2/gemm_register_p8_tn_v1", 90000, 170000),
            ncu_row(7, "g0:register:2/gemm_split_reduce", 3000, 5600),
            ncu_row(8, "g0:cublas:2/ampere_sgemm_64x64_tn", 16200, 30500)])
        calls = parse_csv(text)
        register = calls[("g0", "register")]
        self.assertEqual([c["rep"] for c in register], [0, 1, 2])
        self.assertEqual(register[0]["cycles"], 39600.)
        self.assertEqual(register[0]["kernels"], ["gemm_register_p8_tn_v1", "gemm_split_reduce"])
        got = summarize(register, 3)
        self.assertEqual(got["cycles"], 39600.)  # the median drops the 170k-cycle outlier
        self.assertEqual(got["kernels_per_call"], 2)
        self.assertAlmostEqual(got["ms"], 0.021)
        self.assertEqual(summarize(calls[("g0", "cublas")], 3)["cycles"], 30500.)
        with self.assertRaises(ValueError):
            summarize(register[:2], 3)

    def test_unlabelled_kernels_and_malformed_ranges_are_refused(self):
        for name in ("ampere_sgemm_64x64_tn", "g0:register/gemm_register_p8_tn_v1"):
            with self.assertRaises(ValueError):
                parse_csv(NCU_HEADER + ncu_row(0, name, 1000, 2000))

    def test_cycles_take_precedence_over_event_time(self):
        row = dict(register_ms=2., cublas_ms=1., register_cycles=1100., cublas_cycles=1000.)
        self.assertAlmostEqual(gemm_ratio(row), 1.1)
        self.assertEqual(gemm_ratio(dict(register_ms=2., cublas_ms=1.)), 2.)
        with self.assertRaises(ValueError):
            gemm_ratio(dict(register_ms=2., cublas_ms=1., register_cycles=0., cublas_cycles=1000.))


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
