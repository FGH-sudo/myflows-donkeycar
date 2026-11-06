"""Exact production-entry resume must preserve the point of unfreezing and the BN policy."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_train_distributed import _tub
from tests.test_train_outcome import _train
from MyFlows.utils.checkpoint import list_training_states


class ProgressiveUnfreezeTest(unittest.TestCase):
    def test_resume_crosses_unfreeze_boundary_bitwise_for_both_bn_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tub(root)
            for policy in ("follow", "keep-frozen"):
                extra = ("--seed", "5", "--unfreeze-schedule", "0:4,1:2", "--bn-unfreeze", policy,
                         "--backbone-lr-mult", "0.1", "--state-every", "2", "--keep-states", "20", "--monitor", "off")
                first, full = _train(root, "full-" + policy, *extra)
                self.assertEqual(first.returncode, 0, first.stdout[-1500:] + first.stderr[-1500:])
                result = json.loads((full / "run/result.json").read_text(encoding="utf-8"))
                history = result["transfer"]["freeze_history"]
                self.assertEqual([r["freeze_stages"] for r in history], [4, 2])
                self.assertTrue(result["transfer"]["frozen_unchanged"])
                saved = next(s for s in list_training_states(full / "run/states") if s["step"] == 2)
                second, resumed = _train(root, "resumed-" + policy, *extra, "--resume-state", str(saved["path"]))
                self.assertEqual(second.returncode, 0, second.stdout[-1500:] + second.stderr[-1500:])
                with np.load(full / "ckpt.npz") as a, np.load(resumed / "ckpt.npz") as b:
                    self.assertEqual(set(a.files), set(b.files))
                    for key in a.files:
                        self.assertEqual(a[key].tobytes(), b[key].tobytes(), f"{policy}: {key}")
                final = json.loads((resumed / "ckpt.json").read_text(encoding="utf-8"))
                self.assertEqual(final["training"]["transfer_state"]["level"], 2)
