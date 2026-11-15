"""CPU-only checks with synthetic PNG masks; no benchmark data are bundled."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import evaluate, load_mask


def sample(root, empty=False):
    pred, gt = root/'predictions', root/'ground_truth'
    (pred/'video_01').mkdir(parents=True)
    (gt/'video_01/gt').mkdir(parents=True)
    array = np.zeros((16, 16), dtype=np.uint8)
    if not empty:
        array[4:12, 4:12] = 255
    for folder in (pred/'video_01', gt/'video_01/gt'):
        Image.fromarray(array).save(folder/'00000.png')
    return pred, gt


class EvaluationTests(unittest.TestCase):
    def test_perfect_prediction_and_explicit_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder))
            result = evaluate(pred, gt)
            self.assertEqual((result['frames'], result['videos']), (1, 1))
            self.assertEqual(result['MAE'], 0)
            self.assertAlmostEqual(result['S_m'], 1, places=6)
            self.assertAlmostEqual(result['F_adaptive'], 1)
            self.assertAlmostEqual(result['F_max'], 1)
            self.assertAlmostEqual(result['F_weighted'], 1, places=5)
            self.assertTrue(all(np.isfinite(result[key]) for key in ('S_m', 'E_adaptive', 'MAE')))

    def test_empty_gt_does_not_generate_an_invalid_max_f(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder), empty=True)
            result = evaluate(pred, gt)
            self.assertEqual(result['empty_gt_frames'], 1)
            self.assertIsNone(result['F_max'])
            self.assertEqual(result['F_adaptive'], 0)
            self.assertEqual(result['MAE'], 0)
            json.dumps(result, allow_nan=False)

    def test_corner_target_has_finite_structure_score(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder))
            array = np.zeros((16, 16), dtype=np.uint8)
            array[-1, -1] = 255
            Image.fromarray(array).save(pred/'video_01/00000.png')
            Image.fromarray(array).save(gt/'video_01/gt/00000.png')
            result = evaluate(pred, gt)
            self.assertTrue(np.isfinite(result['S_m']))
            self.assertAlmostEqual(result['S_m'], 1, places=6)

    def test_inverted_prediction_has_nonnegative_structure_score(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder))
            array = np.zeros((16, 16), dtype=np.uint8)
            array[4:12, 4:12] = 255
            Image.fromarray(255-array).save(pred/'video_01/00000.png')
            result = evaluate(pred, gt)
            self.assertGreaterEqual(result['S_m'], 0)
            self.assertEqual(result['MAE'], 1)

    def test_filename_mismatch_is_not_silently_matched_by_order(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder))
            (pred/'video_01/00000.png').rename(pred/'video_01/00001.png')
            with self.assertRaises(ValueError):
                evaluate(pred, gt)

    def test_resizing_requires_explicit_request(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder))
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(pred/'video_01/00000.png')
            with self.assertRaises(ValueError):
                evaluate(pred, gt)
            result = evaluate(pred, gt, resize=True)
            self.assertTrue(result['evaluation']['resize_predictions'])

    def test_unsupported_bit_depth_is_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mask.png'
            array = np.array([[0, 256], [1024, 65535]], dtype=np.uint16)
            Image.fromarray(array).save(path)
            with self.assertRaises(ValueError):
                load_mask(path)

    def test_non_png_images_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            pred, gt = sample(Path(folder))
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(gt/'video_01/gt/00001.jpg')
            with self.assertRaises(ValueError):
                evaluate(pred, gt)

    def test_cli_writes_json_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pred, gt = sample(root)
            output = root/'results/metrics.json'
            command = [sys.executable, '-B', str(ROOT/'evaluate.py'), '--pred-root', str(pred),
                       '--gt-root', str(gt), '--output', str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())['frames'], 1)
            original = output.read_bytes()
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
