from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from apps.common.cifar10 import RECORD_BYTES, _batch, balanced_subset, stratified_split


class CifarBinaryTest(unittest.TestCase):
    def test_planar_pixels_and_file_validation(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.bin"
            record = np.zeros((1, RECORD_BYTES), dtype=np.uint8)
            record[0, 0] = 7
            record[0, 1:1025] = 100
            record.tofile(path)
            x, y = _batch(path, rows=1)
            self.assertEqual(y.tolist(), [7])
            self.assertEqual(x.shape, (1, 3, 32, 32))
            self.assertTrue(np.all(x[0, 0] == 100))
            self.assertTrue(np.all(x[0, 1:] == 0))
            record[0, 0] = 10
            record.tofile(path)
            with self.assertRaisesRegex(ValueError, "labels"):
                _batch(path, rows=1)
            path.write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "bytes"):
                _batch(path, rows=1)

    def test_fixed_balanced_split_and_subset_have_no_cross_split_members(self):
        labels = np.repeat(np.arange(10), 20)
        train, val = stratified_split(labels, val_size=50, seed=13)
        self.assertFalse(np.intersect1d(train, val).size)
        np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(200))
        np.testing.assert_array_equal(np.bincount(labels[val]), np.full(10, 5))
        small = balanced_subset(train, labels, 32, seed=5)
        self.assertTrue(set(small).issubset(train))
        np.testing.assert_array_equal(np.bincount(labels[small]), [4, 4, 3, 3, 3, 3, 3, 3, 3, 3])
        np.testing.assert_array_equal(stratified_split(labels, val_size=50, seed=13)[1], val)


class CifarOverfitTest(unittest.TestCase):
    def test_small_cnn_memorizes_32_balanced_real_images(self):
        source = Path(__file__).resolve().parents[1] / "runs/datasets/cifar10/cifar-10-batches-bin/data_batch_1.bin"
        if not source.is_file():
            self.skipTest("requires local CIFAR-10 binary training data")
        import MyFlows as ms
        from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device
        from apps.train.train_classify import build, evaluate

        previous = get_device()
        try:
            images, labels = _batch(source)
            indices = balanced_subset(np.arange(len(labels)), labels, 32, seed=42)
            model, nodes, _ = build({"seed": 1, "width": 8, "batch": 32, "arch": "small",
                                     "device": "cuda" if cuda_available() else "cpu"})
            x, y, logits, _, graph = nodes
            x.value, y.value = images[indices].astype(np.float32)/255., labels[indices]
            optimizer = ms.Adam(graph, learning_rate=.005)
            for step in range(160):
                optimizer.one_step()
                optimizer.update()
                if step % 8 == 7 and np.mean(asnumpy(logits.value).argmax(axis=1) == labels[indices]) >= .95:
                    break
            score = evaluate(model, nodes, images, labels, indices, 32)["accuracy"]
            self.assertGreaterEqual(score, .95)
        finally:
            set_device(previous)
