"""
gradcam.py
----------
Grad-CAM implementation compatible with ConvNeXt-Tiny, VGG16, and ResNet50.

Usage
-----
    from utils.gradcam import GradCAM, visualise_gradcam

    cam = GradCAM(model, target_layer)
    heatmap = cam(input_tensor, class_idx)          # or class_idx=None → pred class
    visualise_gradcam(original_pil, heatmap, pred_label, save_path)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Generic Grad-CAM that works on any layer that produces a (N, C, H, W) feature map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._features = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self._features = output.detach()

        def bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Returns a (H, W) numpy heatmap in [0, 1].
        input_tensor: (1, C, H, W)  already on the model's device.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool gradients → weights
        grads   = self._gradients      # (1, C, H, W)
        feats   = self._features       # (1, C, H, W)

        # Handle ConvNeXt which outputs (1, C, H, W) with channels-last internally
        if grads.dim() == 4:
            weights = grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        else:
            weights = grads.mean(dim=1, keepdim=True)

        cam = (weights * feats).sum(dim=1).squeeze(0)        # (H, W)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def predict_class(self, input_tensor: torch.Tensor) -> int:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
        return logits.argmax(dim=1).item()


def visualise_gradcam(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    pred_label: str,
    save_path: str,
    image_size: tuple = (256, 256),
    alpha: float = 0.45,
):
    """
    Saves a three-panel figure:
        [Original image] | [Grad-CAM heatmap] | [Overlay]

    Shows that the model focuses on disease-affected lung regions.
    """
    import matplotlib.pyplot as plt

    img = np.array(original_pil.resize(image_size))

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, image_size)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap_rgb    = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, colormap_rgb, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img);         axes[0].set_title("Original Image");     axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet");
    axes[1].set_title("Grad-CAM Heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title(f"Overlay\n{pred_label}"); axes[2].axis("off")

    plt.suptitle(f"Grad-CAM  –  Predicted: {pred_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Helper: pick the right target layer per model architecture ─────────────

def get_target_layer(model, model_name: str):
    """
    Returns the last convolutional / feature layer appropriate for Grad-CAM.
    model_name: 'convnext', 'vgg16', 'resnet50'
    """
    name = model_name.lower()
    if "convnext" in name:
        # Last block of ConvNeXt features
        return model.backbone.features[-1]
    elif "vgg" in name:
        return model.features[-1]          # last MaxPool → use last conv before it
    elif "resnet" in name:
        return model.model.layer4[-1]      # last residual block
    else:
        raise ValueError(f"Unknown model name: {model_name}")
