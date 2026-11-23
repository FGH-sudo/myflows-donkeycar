"""Spatial road regression: train, checkpoint reload and the actual ONNX export agree."""

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.test_train_outcome import _train
from tests.test_train_distributed import _tub


class SpatialRegressionTest(unittest.TestCase):
    def test_training_and_export_preserve_the_spatial_head(self):
        self._check_export("resnet18", 8)

    def test_resnet34_spatial_export_from_uncompressed_checkpoint(self):
        self._check_export("resnet34", 16, "--no-compress-checkpoints")

    def _check_export(self, arch, blocks, *checkpoint_flags):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _tub(work)
            result, output = _train(work, "spatial", "--arch", arch, "--regression-head", "spatial",
                                    "--epochs", "1", "--seed", "2", *checkpoint_flags,
                                    "--export-onnx", env={"MYFLOWS_GPU_LOCK": str(work / "gpu.lock")})
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            self.assertTrue((output / "best.onnx").is_file())
            checkpoint = json.loads((output / "best.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(layer["class"] == "BasicBlock" for layer in checkpoint["model"]["layers"]), blocks)
            with zipfile.ZipFile(output / "ckpt.npz") as archive:
                compression = zipfile.ZIP_STORED if checkpoint_flags else zipfile.ZIP_DEFLATED
                self.assertTrue(all(item.compress_type == compression for item in archive.infolist()))
            self.assertTrue(any(p["shape"][0] == 4 * 8 * 1 * 2 for p in checkpoint["model"]["params"] if p["name"] == "fc_W"))
            script = '''
import json,sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import onnxruntime as ort
import MyFlows as ms
from MyFlows.core.device import set_device
from apps.train.train_myflows_donkey import _build_model
set_device('cpu')
ms.set_default_dtype('float32')
out=Path(sys.argv[1])
cfg=json.loads((out/'run/config.json').read_text(encoding='utf-8'))
x,_,model,logits,_,_=_build_model(Namespace(**cfg['args']),1,cfg['input_h'],cfg['input_w'],np.float32,'numpy',load_pretrained=False)
model.eval()
ms.load_checkpoint([model],None,str(out/'best'))
graph=ms.Graph(logits)
x.value=np.random.default_rng(3).random((1,3,cfg['input_h'],cfg['input_w'])).astype(np.float32)
graph.forward()
sess=ort.InferenceSession(str(out/'best.onnx'),providers=['CPUExecutionProvider'])
actual=sess.run(None,{sess.get_inputs()[0].name:x.value})[0]
np.testing.assert_allclose(actual,logits.value,rtol=2e-5,atol=2e-5)
'''
            checked = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(output)],
                                     cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=90)
            self.assertEqual(checked.returncode, 0, checked.stdout[-2000:] + checked.stderr[-3000:])
