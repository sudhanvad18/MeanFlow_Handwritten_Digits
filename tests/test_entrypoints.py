"""Small regressions for inference imports; no dataset or trained model is needed."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from torchvision import datasets


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EntrypointTests(unittest.TestCase):
    def test_training_import_does_not_load_mnist(self):
        with patch.object(datasets, "MNIST", side_effect=AssertionError("Dataset loaded on import")) as mnist:
            module = load_module("MeanFlow_Train", "MeanFlow_Train.py")
        mnist.assert_not_called()
        self.assertTrue(callable(module.train))

    def test_generation_import_resolves_without_loading_mnist(self):
        with patch.object(datasets, "MNIST", side_effect=AssertionError("Dataset loaded on import")) as mnist:
            load_module("MeanFlow_Train", "MeanFlow_Train.py")
            module = load_module("meanflow_generate_test", "generate.py")
        mnist.assert_not_called()
        self.assertTrue(callable(module.generate_samples))


if __name__ == "__main__":
    unittest.main()
