import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from negative_restoration.data import read_manifest
from negative_restoration.prepare import restoration_variants, color_variants


class DataTests(unittest.TestCase):
    def test_manifest_paths_and_missing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (28, 42)).save(root / "image.png")
            manifest = root / "pairs.json"
            manifest.write_text(json.dumps([{"input": "image.png", "target": "image.png"}]))
            self.assertEqual(read_manifest(manifest, "restoration")[0]["input"], root / "image.png")
            manifest.write_text(json.dumps([{"input": "image.png", "target": "missing.png"}]))
            with self.assertRaises(FileNotFoundError):
                read_manifest(manifest, "restoration")

    def test_paired_augmentation_geometry(self):
        rng = np.random.default_rng(5)
        image = rng.integers(0, 255, (530, 560, 3), dtype=np.uint8)
        restoration = list(restoration_variants(image, image.copy(), True))
        self.assertEqual(len(restoration), 16)
        for inp, target in restoration:
            self.assertEqual(inp.shape, (504, 504, 3))
            np.testing.assert_array_equal(inp, target)
        reference = image[:450, :430]
        color = list(color_variants(image, reference, image.copy(), "sample", 123, True))
        repeated = list(color_variants(image, reference, image.copy(), "sample", 123, True))
        self.assertEqual(len(color), 4)
        for (inp, ref, target), again in zip(color, repeated):
            self.assertEqual(ref.shape, (512, 512, 3))
            np.testing.assert_array_equal(inp, target)
            for a, b in zip((inp, ref, target), again):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
