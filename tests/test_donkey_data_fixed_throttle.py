import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from apps.common.donkey_data import load_donkey_index


class DonkeyDataFixedThrottleTest(unittest.TestCase):
    def test_force_fixed_throttle_overrides_catalog_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            images_dir = data_dir / "images"
            images_dir.mkdir()
            cv2.imwrite(str(images_dir / "0_0.1000.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
            record = {
                "cam/image_array": "0_0.1000.jpg",
                "user/angle": 0.1,
                "user/throttle": 0.5,
            }
            (data_dir / "catalog_generated.catalog").write_text(json.dumps(record) + "\n", encoding="utf-8")

            normal = load_donkey_index(data_dir, 0.2, 1.0)
            forced = load_donkey_index(data_dir, 0.2, 1.0, force_fixed_throttle=True)

        self.assertEqual(normal[0][2], 0.5)
        self.assertEqual(forced[0][2], 0.2)


if __name__ == "__main__":
    unittest.main()
