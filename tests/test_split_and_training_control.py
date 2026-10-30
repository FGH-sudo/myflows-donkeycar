import json
import tempfile
import unittest
from pathlib import Path

from apps.common.splits import (
    build_split,
    load_split,
    resolve_splits,
    save_split,
    select_split,
)
from apps.train.common.training_control import EarlyStopping


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
