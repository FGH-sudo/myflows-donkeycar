"""P9 TrainingState: transactional checksummed states, bitwise-exact resume, fingerprint checks, lr override."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MyFlows.utils.checkpoint import TrainingStateError, list_training_states, verify_training_state
from tests.test_train_distributed import _tub
from tests.test_train_outcome import _params, _train

NOISY = ("--seed", "5", "--augment", "--dropout", "0.3", "--mixup", "--state-every", "2", "--keep-states", "50")


class TrainingStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        _tub(cls.work)
        result, cls.full = _train(cls.work, "full", *NOISY)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        cls.states = list_training_states(cls.full / "run" / "states")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def resume(self, name, state, *extra):
        return _train(self.work, name, *NOISY, "--resume-state", str(state["path"]), *extra)

    def test_states_are_complete_and_checksummed(self):
        self.assertGreaterEqual(len(self.states), 3)
        latest = json.loads((self.full / "run" / "states" / "latest.json").read_text(encoding="utf-8"))
        self.assertEqual(latest["step"], self.states[-1]["step"])
        for item in self.states:
            manifest = verify_training_state(item["path"])
            self.assertEqual(set(manifest["files"]), {"checkpoint.json", "checkpoint.npz"})
            state = item["state"]
            self.assertEqual(len(state["rng"]["transforms"]), 3)
            self.assertTrue(state["rng"]["exact"])
            self.assertEqual(state["global_step"], item["step"])
        self.assertTrue(any(not s["state"]["epoch_complete"] for s in self.states))
        self.assertTrue(any(s["state"]["epoch_complete"] for s in self.states))
        self.assertFalse(list((self.full / "run" / "states").glob(".staging-*")))

    def test_resume_from_mid_epoch_and_epoch_end_is_bitwise_identical(self):
        full = _params(self.full)
        start = self.states[0]
        self.assertEqual((start["step"], start["state"]["next_step"], start["state"]["trained_samples"]), (0, 0, 0),
                         "a fresh run saves its initial weights as step 0")
        mid = next(s for s in self.states if not s["state"]["epoch_complete"] and s["state"]["epoch"] == 0
                   and s["step"] > 0)
        end = next(s for s in self.states if s["state"]["epoch_complete"] and s["state"]["epoch"] == 0)
        for name, state in (("from-start", start), ("from-mid", mid), ("from-end", end)):
            result, out = self.resume(name, state)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            resumed = _params(out)
            for key, value in full.items():
                np.testing.assert_array_equal(value, resumed[key], err_msg=f"{name}: {key}")
            report = json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))
            original = json.loads((self.full / "run" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["samples"]["trained"], report["samples"]["expected_trained"])
            self.assertEqual(report["samples"]["expected_trained"], original["samples"]["expected_trained"])
            self.assertEqual(report["metrics"]["val_angle_mae"], original["metrics"]["val_angle_mae"])
            self.assertEqual(report["resume"]["from_global_step"], state["step"])

    def test_config_change_and_corruption_are_refused(self):
        state = self.states[0]
        result, _ = self.resume("other-batch", state, "--batch", "4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("batch", result.stdout + result.stderr)
        broken = self.work / "broken" / state["path"].name
        broken.parent.mkdir()
        import shutil

        shutil.copytree(state["path"], broken)
        data = bytearray((broken / "checkpoint.npz").read_bytes())
        data[len(data) // 2] ^= 0xFF
        (broken / "checkpoint.npz").write_bytes(bytes(data))
        with self.assertRaises(TrainingStateError):
            verify_training_state(broken)
        result, _ = _train(self.work, "corrupt", *NOISY, "--resume-state", str(broken))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("校验和", result.stdout + result.stderr)

    def test_lr_override_is_recorded(self):
        state = self.states[1]
        result, out = self.resume("half-lr", state, "--lr", "0.0005")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        report = json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((report["resume"]["lr_before"], report["resume"]["lr"]), (0.001, 0.0005))
        later = list_training_states(out / "run" / "states")[-1]["state"]
        self.assertEqual(later["lr_history"][-1], {"global_step": state["step"], "lr": 0.0005})
        self.assertEqual(later["lr"], 0.0005)


class PruningTest(unittest.TestCase):
    def test_the_initial_state_survives_pruning(self):
        from MyFlows.utils.checkpoint import save_training_state

        with tempfile.TemporaryDirectory() as tmp:
            for step in range(0, 6):
                save_training_state(tmp, [], None, step, {"epoch": 0, "global_step": step}, keep=2)
            self.assertEqual([s["step"] for s in list_training_states(tmp)], [0, 4, 5])



class FingerprintCompatibilityTest(unittest.TestCase):
    def test_transfer_arguments_at_their_defaults_leave_the_fingerprint_unchanged(self):
        from apps.train.train_myflows_donkey import build_parser, trajectory_config

        default = trajectory_config(build_parser().parse_args([]))
        self.assertFalse({"arch", "pretrained", "freeze_stages", "image_size", "input_norm"} & set(default))
        transfer = trajectory_config(build_parser().parse_args(["--arch", "mobilenet_v2", "--freeze-stages", "2"]))
        self.assertEqual((transfer["arch"], transfer["freeze_stages"]), ("mobilenet_v2", 2))


if __name__ == "__main__":
    unittest.main()
