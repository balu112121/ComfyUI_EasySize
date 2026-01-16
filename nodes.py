# nodes.py
import torch
import numpy as np
from PIL import Image, ImageOps
from .presets import PRESETS, CROP_METHODS, RESIZE_ALGOS, get_size_from_preset

# ------------- 通用工具 -------------
def resize_crop(image: Image.Image, tgt_w, tgt_h, crop_method, algo) -> Image.Image:
    if crop_method == "中心裁剪":
        image = ImageOps.fit(image, (tgt_w, tgt_h), method=Image.Resampling[algo.upper()])
    else:
        image = image.resize((tgt_w, tgt_h), resample=Image.Resampling[algo.upper()])
    return image

def resize_by_long_or_short(pil_img: Image.Image, edge_mode: str, target_len: int) -> Image.Image:
    w, h = pil_img.size
    if edge_mode == "最长边":
        if w >= h:
            new_w, new_h = target_len, int(h * target_len / w)
        else:
            new_w, new_h = int(w * target_len / h), target_len
    else:  # 最短边
        if w <= h:
            new_w, new_h = target_len, int(h * target_len / w)
        else:
            new_w, new_h = int(w * target_len / h), target_len
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

# ------------- 简单图像尺寸 -------------
class EasySizeSimpleImage:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_dict = {k: ["关"] + [t[0] for t in PRESETS[k]] for k in PRESETS}
        return {
            "required": {
                **{k: (v, {"default": "关"}) for k, v in preset_dict.items()},
                "裁剪方式": (CROP_METHODS, {"default": "中心裁剪"}),
                "缩放算法": (RESIZE_ALGOS, {"default": "lanczos"}),
                "启用边长缩放": ("BOOLEAN", {"default": False}),
                "缩放至边": (["最长边", "最短边"], {"default": "最长边"}),
                "缩放长度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("图像", "遮罩", "宽", "高")
    FUNCTION = "run"
    CATEGORY = "EasySize"

    def run(self, 图像=None, 遮罩=None, **kwargs):
        use_edge = kwargs["启用边长缩放"]
        edge_mode = kwargs["缩放至边"]
        target_len = kwargs["缩放长度"]
        crop = kwargs["裁剪方式"]
        algo = kwargs["缩放算法"]

        if use_edge:
            # 仅用边长缩放，预设完全失效
            if 图像 is not None:
                b, h0, w0, c = 图像.shape
                arr = (图像.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
                pil_img = resize_by_long_or_short(pil_img, edge_mode, target_len)
                arr = np.array(pil_img).astype(np.float32) / 255.0
                图像 = torch.from_numpy(arr).unsqueeze(0)
                out_w, out_h = pil_img.size
            else:
                图像 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                out_w, out_h = 512, 512

            if 遮罩 is not None:
                b, h0, w0 = 遮罩.shape
                arr = (遮罩.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pil_msk = Image.fromarray(arr, mode="L")
                pil_msk = resize_by_long_or_short(pil_msk, edge_mode, target_len)
                arr = np.array(pil_msk).astype(np.float32) / 255.0
                遮罩 = torch.from_numpy(arr).unsqueeze(0)
            else:
                遮罩 = torch.zeros((1, out_h, out_w), dtype=torch.float32)

            return (图像, 遮罩, out_w, out_h)

        # 原逻辑：预设 + 裁剪/缩放
        choices = {k: kwargs[k] for k in PRESETS}
        w, h = get_size_from_preset(choices)

        if 图像 is not None:
            b, h0, w0, c = 图像.shape
            arr = (图像.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
            pil_img = resize_crop(pil_img, w, h, crop, algo)
            arr = np.array(pil_img).astype(np.float32) / 255.0
            图像 = torch.from_numpy(arr).unsqueeze(0)
        else:
            图像 = torch.zeros((1, h, w, 3), dtype=torch.float32)

        if 遮罩 is not None:
            b, h0, w0 = 遮罩.shape
            arr = (遮罩.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pil_msk = Image.fromarray(arr, mode="L")
            pil_msk = resize_crop(pil_msk, w, h, crop, algo)
            arr = np.array(pil_msk).astype(np.float32) / 255.0
            遮罩 = torch.from_numpy(arr).unsqueeze(0)
        else:
            遮罩 = torch.zeros((1, h, w), dtype=torch.float32)

        return (图像, 遮罩, w, h)

# ------------- 简单图像尺寸-Latent -------------
class EasySizeSimpleLatent:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_dict = {k: ["关"] + [t[0] for t in PRESETS[k]] for k in PRESETS}
        return {
            "required": {
                **{k: (v, {"default": "关"}) for k, v in preset_dict.items()},
                "启用自定义尺寸": ("BOOLEAN", {"default": False}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "EasySize"

    def run(self, **kwargs):
        use_custom = kwargs["启用自定义尺寸"]
        if use_custom:
            w, h = kwargs["宽度"], kwargs["高度"]
        else:
            choices = {k: kwargs[k] for k in PRESETS}
            w, h = get_size_from_preset(choices)
        latent = torch.zeros([1, 4, h // 8, w // 8])
        return ({"samples": latent},)

# ------------- 注册 -------------
NODE_CLASS_MAPPINGS = {
    "EasySizeSimpleImage":  EasySizeSimpleImage,
    "EasySizeSimpleLatent": EasySizeSimpleLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasySizeSimpleImage":  "简单图像尺寸",
    "EasySizeSimpleLatent": "简单图像尺寸-Latent",
}