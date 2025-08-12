from __future__ import annotations

import unittest

from circuit_reuse.dataset import AdditionDataset, ArithmeticExample
from circuit_reuse.dataset import MMLUDataset, MIBDatasetHF
import importlib
from circuit_reuse.circuit_extraction import Component, compute_shared_circuit
from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_ablation
from .mock_model import MockModel

class TestCircuitReuse(unittest.TestCase):
    def test_compute_shared_circuit(self):
        # two circuits with partial overlap
        c1 = {Component(0, "head", 0), Component(1, "mlp", 0)}
        c2 = {Component(0, "head", 0), Component(2, "mlp", 0)}
        shared = compute_shared_circuit([c1, c2])
        self.assertEqual(shared, {Component(0, "head", 0)})

    def test_evaluate_accuracy(self):
        # create a dummy dataset of two examples, both should be predicted as token 0
        ds = [
            ArithmeticExample(prompt="1 + 1 =", target="0"), 
            ArithmeticExample(prompt="2 + 2 =", target="0")
        ]
        predictions = {
            "1 + 1 =": 0, 
            "2 + 2 =": 0
        }
        
        model = MockModel(predictions)
        
        # manually set prediction before each call (simulate internal logic)
        accs = []
        for ex in ds:
            model.set_prediction_for_prompt(ex.prompt)
            pred = evaluate_accuracy(model, [ex], "arithmetic", mock=True)
            accs.append(pred)
        self.assertEqual(accs, [1.0, 1.0])
        
        # evaluate entire dataset with ablation (should still be correct)
        for ex in ds:
            model.set_prediction_for_prompt(ex.prompt)
        acc_ablate = evaluate_accuracy_with_ablation(model, ds, "arithmetic", removed=[], mock=True)  # no removal
        self.assertEqual(acc_ablate, 1.0)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets library not available")
    def test_mmlu_dataset_loading(self):
        # load a small number of examples from a  MMLU subject
        ds = MMLUDataset(subject="abstract_algebra", split="test", num_examples=3)
        
        # ensure at least one example is loaded and has both prompt and target
        self.assertGreater(len(ds), 0)
        for ex in ds:
            self.assertIsInstance(ex.prompt, str)
            self.assertIsInstance(ex.target, str)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets library not available")
    def test_mib_dataset_loading(self):
        # load a small number of examples from a MIB dataset (ioi)
        ds = MIBDatasetHF(name="ioi", split="test", num_examples=3)
        
        # ensure some examples loaded and prompt/target are non-empty strings
        self.assertGreater(len(ds), 0)
        for ex in ds:
            self.assertIsInstance(ex.prompt, str)
            self.assertIsInstance(ex.target, str)

if __name__ == "__main__":
    unittest.main()