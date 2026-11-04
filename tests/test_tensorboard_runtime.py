"""The training dashboard must write TensorBoard events without loading torch's CUDA DLLs."""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "requires tensorboard")
class TensorBoardRuntimeTest(unittest.TestCase):
    def test_events_roundtrip_in_process_without_torch(self):
        script = r'''
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from MyFlows.utils.tensorboard_logger import TensorBoardLogger
with TensorBoardLogger(sys.argv[1]) as logger:
    logger.log_scalar("resources/power", 25., 7, wall_time=123.5)
    logger.log_text("config/training_args", '{"batch": 2}')
    logger.log_histogram("weights", np.array([-2., 0., 1.]), 7)
    logger.log_images_grid("images", [np.zeros((3, 8, 8)), np.ones((3, 8, 8))], 7)
    logger.log_scalar_dict("phases", {"forward": 1., "backward": 2.}, 7)
assert "torch" not in sys.modules, "runtime imported torch"
events = EventAccumulator(sys.argv[1]).Reload()
scalar = events.Scalars("resources/power")[0]
assert (scalar.step, scalar.value, scalar.wall_time) == (7, 25., 123.5)
assert events.Tensors("config/training_args/text_summary")[0].tensor_proto.string_val == [b'{"batch": 2}']
assert events.Histograms("weights")[0].histogram_value.num == 3
assert len(events.Images("images")) == 1
'''
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run([sys.executable, "-X", "utf8", "-c", script, tmp],
                                     cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                                     encoding="utf-8", timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
