import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from negative_restoration.io import build_model, load_checkpoint
from negative_restoration.train import loss_fn, resume_training, save_training


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self, task):
        inp = torch.rand(1, 3, 29, 43)
        if task == "restoration":
            return inp, torch.randn(1, 3, 4, 4, 16)
        return inp, torch.rand(1, 3, 43, 29), torch.randn(1, 16, 3, 4), torch.randn(1, 16, 4, 3)

    def test_forward_backward_and_resume(self):
        for task in ("restoration", "color_mapping"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(7)
                model = build_model(task)
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
                args = self.inputs(task)
                target = torch.rand_like(args[0])
                train_cfg = {"fft_weight": 0.125}
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(*args)
                    self.assertEqual(pred.shape, target.shape)
                    loss = loss_fn(pred, target, task, train_cfg)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    optimizer.step()
                if task == "restoration":
                    self.assertIsNotNone(model.n_logit.grad)
                    self.assertGreater(abs(model.n_logit.grad.item()), 0)
                else:
                    self.assertGreater(model.backbone.res_align.mlp[0].weight.grad.abs().sum().item(), 0)
                checkpoint = Path(directory) / "last.pt"
                save_training(checkpoint, model, optimizer, 2, {"task": task})
                restored = build_model(task)
                opt2 = torch.optim.AdamW(restored.parameters(), lr=2e-4, weight_decay=0)
                self.assertEqual(resume_training(checkpoint, restored, opt2, task), 2)
                with torch.no_grad():
                    torch.testing.assert_close(model(*args), restored(*args), rtol=0, atol=0)
                # Both optimizer states must produce the same subsequent update.
                for net, opt in ((model, optimizer), (restored, opt2)):
                    opt.zero_grad(set_to_none=True)
                    loss_fn(net(*args), target, task, train_cfg).backward()
                    opt.step()
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_legacy_numpy_scalar_and_strict_loading(self):
        model = build_model("color_mapping")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"model": model.state_dict(), "lr": np.float64(1e-8)}, path)
            load_checkpoint(build_model("color_mapping"), path)
            state = model.state_dict()
            state.pop("backbone.intro.weight")
            torch.save({"model": state}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, path)

    def test_bad_restoration_grid_is_rejected(self):
        model = build_model("restoration")
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 28, 28), torch.rand(1, 3, 5, 5, 16))


if __name__ == "__main__":
    unittest.main()
