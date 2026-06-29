import csv
import tempfile
import unittest
from pathlib import Path

from benchmark import dataloader_bench


class DataloaderBenchOutputsTest(unittest.TestCase):
    def test_enrich_and_write_results(self):
        rows = [
            {"num_workers": 0, "batches": 2, "batch_size": 4, "elapsed_s": 2.0, "img_s": 4.0},
            {"num_workers": 2, "batches": 2, "batch_size": 4, "elapsed_s": 1.0, "img_s": 8.0},
        ]

        enriched = dataloader_bench.enrich_results(rows)
        self.assertEqual(enriched[0]["speedup"], 1.0)
        self.assertEqual(enriched[1]["speedup"], 2.0)
        self.assertEqual(enriched[1]["ms_per_batch"], 500.0)

        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "bench.csv"
            out_md = Path(tmp) / "bench.md"
            dataloader_bench.write_csv(enriched, out_csv)
            dataloader_bench.write_markdown(enriched, out_md)

            with out_csv.open(encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]["num_workers"], "0")
            self.assertIn("img_s", saved[0])
            self.assertIn("speedup", saved[0])
            self.assertIn("train_ms", saved[0])
            self.assertIn("| num_workers |", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
