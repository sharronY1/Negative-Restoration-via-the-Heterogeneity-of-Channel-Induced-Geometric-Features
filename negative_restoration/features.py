"""Frozen DA3 L13 features with task-specific image geometry."""

import os

import numpy as np
import torch


def pad_image(rgb, multiple):
    h, w = rgb.shape[:2]
    return np.pad(rgb, ((0, -h % multiple), (0, -w % multiple), (0, 0)), mode="edge")


class DA3Features:
    def __init__(self, model_id="depth-anything/DA3NESTED-GIANT-LARGE", device="cuda"):
        os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(model_id).to(device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def _extract(self, views, resolution):
        pred = self.model.inference(
            image=views, export_feat_layers=[13], export_dir=None,
            process_res=resolution, process_res_method="upper_bound_resize",
            ref_view_strategy="first",
        )
        if pred.aux is None or "feat_layer_13" not in pred.aux:
            raise RuntimeError("DA3 did not return L13 features")
        features = np.asarray(pred.aux["feat_layer_13"])
        while features.ndim > 4 and features.shape[0] == 1:
            features = features[0]
        if features.ndim == 3 and len(views) == 1:
            features = features[None]
        if features.ndim != 4 or features.shape[0] != len(views):
            raise RuntimeError(f"Unexpected DA3 feature shape: {features.shape}")
        return features.astype(np.float16), pred.processed_images

    def restoration(self, rgb):
        canvas = pad_image(rgb, 28)
        views = [np.repeat(canvas[:, :, c:c+1], 3, axis=2) for c in range(3)]
        features, processed = self._extract(views, max(canvas.shape[:2]))
        expected = (canvas.shape[0] // 14, canvas.shape[1] // 14)
        if features.shape[1:3] != expected or np.asarray(processed[0]).shape[:2] != canvas.shape[:2]:
            raise RuntimeError("DA3 changed the restoration canvas geometry")
        return canvas, features

    def color(self, rgb, *, training=False):
        # Historical crop extraction resizes the 512 crop to 518; NAF still
        # receives the original crop. Full-image inference uses pad-only /14.
        canvas = rgb if training else pad_image(rgb, 14)
        features, _ = self._extract([canvas], 518 if training else max(canvas.shape[:2]))
        return canvas, features[0].transpose(2, 0, 1).copy()
