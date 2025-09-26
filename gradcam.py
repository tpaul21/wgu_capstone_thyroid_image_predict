# File: project/gradcam.py  (REPLACE)
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer="layer4", forward_fn=None):
        """
        model: torch.nn.Module
        target_layer: name or suffix of the target conv block (e.g., 'layer4' or 'backbone.layer4')
        forward_fn: optional callable that takes (x_img) and returns logits; defaults to model(x_img)
                    Use this for multimodal models: forward_fn = lambda x: model(x, x_tab)
        """
        self.model = model.eval()
        self.forward_fn = forward_fn if forward_fn is not None else (lambda x: model(x))

        # Resolve target layer by suffix match
        modules = dict(model.named_modules())
        if target_layer in modules:
            self.target = modules[target_layer]
        else:
            matches = [m for n, m in modules.items() if n.endswith(target_layer)]
            if not matches:
                raise KeyError(f"Target layer '{target_layer}' not found in model modules: "
                               f"{list(modules.keys())[:10]} ...")
            self.target = matches[-1]  # last suffix match

        self.gradients = None
        self.activations = None
        self.target.register_forward_hook(self._fwd_hook)
        # full_backward hook for new PyTorch; fallback if unavailable
        if hasattr(self.target, "register_full_backward_hook"):
            self.target.register_full_backward_hook(self._bwd_hook)
        else:
            self.target.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, out):
        self.activations = out.detach()

    def _bwd_hook(self, _, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x_img, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.forward_fn(x_img).squeeze(1)
        if class_idx is None:
            class_idx = int(torch.sigmoid(logits).item() >= 0.5)
        score = logits if class_idx == 1 else -logits
        score.backward(retain_graph=True)

        grads = self.gradients   # [B,C,h,w]
        acts  = self.activations # [B,C,h,w]
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=x_img.shape[-2:], mode='bilinear', align_corners=False)[0,0]
        return cam.detach().cpu().numpy()

def overlay_cam(pil_img, cam, alpha=0.35):
    img = np.array(pil_img.resize((cam.shape[1], cam.shape[0]))).astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    heat = plt.cm.jet(cam)[..., :3] * 255.0
    return ((1 - alpha) * img + alpha * heat).astype(np.uint8)
