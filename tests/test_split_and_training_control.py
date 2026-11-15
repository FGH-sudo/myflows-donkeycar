import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apps.common.donkey_data import image_rel_path, parse_angle_from_filename
from apps.common.splits import (
    build_split,
    load_split,
    resolve_splits,
    save_split,
    select_split,
)
from apps.train.common.training_control import BestScore, EarlyStopping
from apps.train.common.validation import run_loss_validation


class SplitAndTrainingControlTest(unittest.TestCase):
    def test_split_is_deterministic_and_size_overrides_ratio(self):
        rows = list(range(20))
        split_a = build_split(rows, val_ratio=0.5, test_ratio=0.5, val_size=3, test_size=4, seed=11)
        split_b = build_split(rows, val_ratio=0.2, test_ratio=0.2, val_size=3, test_size=4, seed=11)

        self.assertEqual(split_a, split_b)
        self.assertEqual(len(split_a["val"]), 3)
        self.assertEqual(len(split_a["test"]), 4)
        self.assertEqual(len(split_a["train"]), 13)
        self.assertEqual(sorted(split_a["train"] + split_a["val"] + split_a["test"]), rows)

    def test_split_roundtrip_and_select(self):
        rows = [f"row{i}" for i in range(8)]
        split = build_split(rows, val_size=2, test_size=1, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            save_split(path, split, source_count=len(rows), seed=5)
            loaded = load_split(path)

        self.assertEqual(loaded["splits"], split)
        self.assertEqual(select_split(rows, loaded["splits"], "val"), [rows[i] for i in split["val"]])
        self.assertEqual(select_split(rows, loaded["splits"], "all"), rows)

    def test_early_stopping_patience_and_min_delta(self):
        stopper = EarlyStopping(patience=2, min_delta=0.1)

        self.assertTrue(stopper.update(1.0).improved)
        self.assertFalse(stopper.update(0.95).should_stop)
        second = stopper.update(0.94)
        self.assertTrue(second.should_stop)
        self.assertEqual(second.best, 1.0)

        improved = stopper.update(0.80)
        self.assertTrue(improved.improved)
        self.assertFalse(improved.should_stop)

    def test_scores_and_split_requests_must_be_finite_and_well_typed(self):
        # NaN compares false with everything: it would never count as an improvement nor as a bad epoch.
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                EarlyStopping(patience=2).update(bad)
            with self.assertRaises(ValueError):
                BestScore().update(bad)
            with self.assertRaises(ValueError):
                EarlyStopping(patience=2, min_delta=bad)
        with self.assertRaises(ValueError):
            EarlyStopping(patience=2, min_delta=-0.1)
        tracker = BestScore()
        self.assertEqual([tracker.update(v) for v in (0.5, 0.6, 0.4)], [True, False, True])
        for kwargs in ({"val_ratio": 1.5}, {"val_ratio": -0.1}, {"val_ratio": float("nan")}, {"test_ratio": 2},
                       {"val_size": -1}, {"val_size": 2.5}, {"test_size": True}, {"val_size": "3"}):
            with self.assertRaises(ValueError, msg=kwargs):
                build_split(10, **kwargs)
        self.assertEqual(len(build_split(10, val_size=0, val_ratio=0.2)["val"]), 2, "size 0 falls back to the ratio")
        self.assertEqual(len(build_split(10, val_size=np.int64(3))["val"]), 3)

    def test_validation_mean_weights_batches_by_their_real_samples(self):
        # Three samples per batch, the last batch holds one: the mean is over the 7 samples, not the 3 batches.
        losses = iter([1.0, 1.0, 8.0])
        counts = iter([3, 3, 1])
        result = run_loss_validation(list(range(7)), 3, load_fn=lambda item: (item, item),
                                     step_fn=lambda xs, ys: (next(losses), next(counts)))
        self.assertEqual((result.batches, result.samples), (3, 7))
        self.assertAlmostEqual(result.mean_loss, (3 * 1.0 + 3 * 1.0 + 8.0) / 7)
        self.assertIsNone(run_loss_validation([], 3, load_fn=lambda item: (item, item), step_fn=lambda xs, ys: (0., 1)))
        for count in (0, -1, 4):
            with self.assertRaises(ValueError, msg=count):
                run_loss_validation(list(range(3)), 3, load_fn=lambda item: (item, item),
                                    step_fn=lambda xs, ys: (1.0, count))

    def test_donkey_index_refuses_paths_outside_images_and_non_finite_labels(self):
        self.assertEqual(image_rel_path("images\\7_0.1.jpg").as_posix(), "images/7_0.1.jpg")
        self.assertEqual(image_rel_path("somewhere/else/7_0.1.jpg").as_posix(), "images/7_0.1.jpg")
        self.assertEqual(image_rel_path("../../secret/7_0.1.jpg").as_posix(), "images/7_0.1.jpg")
        for escaping in ("images/../../secret.jpg", "images\\..\\x.jpg"):
            with self.assertRaises(ValueError, msg=escaping):
                image_rel_path(escaping)
        self.assertEqual(parse_angle_from_filename("12_−0,25.jpg"), -0.25)
        for name in ("3_nan.jpg", "3_inf.jpg", "3_-inf.jpg"):
            with self.assertRaises(ValueError, msg=name):
                parse_angle_from_filename(name)

    def test_loaded_splits_reject_overlap_duplicates_and_invalid_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            for splits in (
                {"train": [0, 1], "val": [1]},
                {"train": [0], "val": [1], "test": [0]},
                {"train": [0, 0], "val": [1]},
                {"train": [-1], "val": [1]},
                {"train": [4], "val": [1]},
                {"train": [0.5], "val": [1]},
                {"train": [True], "val": [1]},
            ):
                with self.subTest(splits=splits):
                    path.write_text(json.dumps({"version": 1, "source_count": 4, "splits": splits}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_split(path)
            save_split(path, {"train": [0, 1], "val": [2]}, source_count=4, seed=0)
            self.assertEqual(load_split(path, source_count=4)["splits"]["train"], [0, 1])
            with self.assertRaisesRegex(ValueError, "source_count"):
                resolve_splits(list(range(5)), split_file=path)


if __name__ == "__main__":
    unittest.main()
