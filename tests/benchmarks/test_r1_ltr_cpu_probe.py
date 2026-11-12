"""Unit tests for the offline probe, independent of torch and model files."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[2] / 'tools/r1_ltr_v1/official_cpu_probe.py'
spec = importlib.util.spec_from_file_location('ltr_cpu_probe', path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_descending_and_ties(self):
        self.assertEqual(probe.stable_descending([-1., 2., 2., 0.]), [1, 2, 3, 0])
        self.assertEqual(probe.stable_descending([0.]), [0])

    def test_invalid_scores(self):
        for x in ([], [float('nan')], [float('inf')], [True], ['1']):
            with self.subTest(scores=x), self.assertRaises(ValueError):
                probe.stable_descending(x)

    def test_hash_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'data'
            p.write_bytes(b'abc')
            self.assertEqual(probe.sha256_file(p), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__ == '__main__':
    unittest.main()
