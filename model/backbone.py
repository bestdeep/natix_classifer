# model/backbone.py
from typing import Tuple, Callable
import inspect
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

# ----------------------------
# Helpers to support both torchvision and timm backbones
# ----------------------------
def _torchvision_pretrained_kwargs(pretrained: bool):
    """
    Return kwargs compatible with the installed torchvision version.
    Newer torchvision uses `weights=...` where weights is an enum, older uses `pretrained=`.
    We attempt to call with `weights` first (if available), else `pretrained`.
    """
    # try to detect new API by signature inspection (weights in signature)
    sig = inspect.signature(tv_models.__dict__.get("mobilenet_v2", lambda **k: None))
    if "weights" in sig.parameters:
        return {"weights": "IMAGENET1K_V1"} if pretrained else {"weights": None}
    return {"pretrained": pretrained}


def _find_last_linear_and_replace(module: nn.Module, num_classes: int) -> Tuple[int, nn.Module]:
    """
    Search the given module for the last nn.Linear (walk reversed order for Sequential),
    return (in_features, modified_module) where modified_module has the linear replaced.

    If num_classes == 0: replace with nn.Identity() so forward returns features.
    If num_classes > 0: replace with nn.Linear(in_features, num_classes).

    Raises RuntimeError if no Linear found.
    """
    # If module itself is a Linear
    if isinstance(module, nn.Linear):
        in_f = module.in_features
        new = nn.Identity() if num_classes == 0 else nn.Linear(in_f, num_classes)
        return in_f, new

    # If Sequential: find last Linear from the end
    if isinstance(module, nn.Sequential):
        for i in range(len(module) - 1, -1, -1):
            if isinstance(module[i], nn.Linear):
                in_f = module[i].in_features
                module[i] = nn.Identity() if num_classes == 0 else nn.Linear(in_f, num_classes)
                return in_f, module
            # nested sequential support
            if isinstance(module[i], nn.Sequential):
                try:
                    in_f, new_sub = _find_last_linear_and_replace(module[i], num_classes)
                    module[i] = new_sub
                    return in_f, module
                except RuntimeError:
                    pass
        # no linear in the sequential
    # Try to inspect children and replace the first matching attribute referencing a Linear
    for name, child in reversed(list(module.named_children())):
        # Try on child recursively
        try:
            in_f, new_child = _find_last_linear_and_replace(child, num_classes)
            # replace the attribute on parent
            setattr(module, name, new_child)
            return in_f, module
        except RuntimeError:
            continue

    raise RuntimeError("Couldn't find a final nn.Linear in the provided module.")


def _prepare_torchvision_model(model: nn.Module, num_classes: int) -> Tuple[nn.Module, int]:
    """
    Given a torchvision model instance, replace its final head according to num_classes.
    Returns (model, feat_dim).
    """
    # Try common attribute names in order of popularity
    attr_candidates = ["classifier", "fc", "head"]

    for attr in attr_candidates:
        if hasattr(model, attr):
            try:
                head = getattr(model, attr)
                in_f, new_head = _find_last_linear_and_replace(head, num_classes)
                setattr(model, attr, new_head)
                return model, in_f
            except RuntimeError:
                # try next candidate
                pass

    # Some torchvision models use .classifier which is a Module but Linear may be nested
    # As a final fallback, search entire model children for last Linear and replace in-place.
    try:
        in_f, _ = _find_last_linear_and_replace(model, num_classes)
        # _find_last_linear_and_replace performs replacement in-place, so return model.
        return model, in_f
    except RuntimeError:
        raise RuntimeError("Unsupported torchvision backbone: cannot locate classifier linear layer.")


def _build_timm_model(name: str, num_classes: int, pretrained: bool):
    """
    Create a timm model. For num_classes==0 we want feature extractor mode that returns pooled features;
    timm supports num_classes=0 and has .num_features attribute.
    """
    if num_classes > 0:
        m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        feat_dim = getattr(m, "num_features", None)
        if feat_dim is None:
            # try to detect final linear
            try:
                _, _ = _find_last_linear_and_replace(m, num_classes)  # no-op replacement
                feat_dim = next((p.in_features for p in m.modules() if isinstance(p, nn.Linear)), None)
            except RuntimeError:
                feat_dim = None
        if feat_dim is None:
            raise RuntimeError(f"Unable to determine feature dim for timm model {name}")
        return m, int(feat_dim)
    else:
        # feature extractor
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = getattr(m, "num_features", None)
        if feat_dim is None:
            # fallback: try to detect last Linear BEFORE we replaced (some timm models may not expose num_features)
            try:
                # Note: this will attempt to find a linear and replace with Identity, but timm models created with num_classes=0
                # usually already return features; still, detect num_features conservatively.
                _, _ = _find_last_linear_and_replace(m, 0)
            except RuntimeError:
                pass
            feat_dim = getattr(m, "num_features", None)
        if feat_dim is None:
            # final fallback: search for any Linear and take its in_features
            for mod in m.modules():
                if isinstance(mod, nn.Linear):
                    feat_dim = mod.in_features
                    break
        if feat_dim is None:
            raise RuntimeError(f"Unable to determine feature dim for timm model {name} (num_classes=0)")
        return m, int(feat_dim)


# ----------------------------
# Concrete builders (torchvision + timm wrappers)
# ----------------------------
def build_model_resnet18(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.resnet18(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_resnet50(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.resnet50(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_mobilenet_v2(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.mobilenet_v2(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_mobilenet_v3_small(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.mobilenet_v3_small(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_mobilenet_v3_large(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.mobilenet_v3_large(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_efficientnet_b0(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.efficientnet_b0(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_efficientnet_b3(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.efficientnet_b3(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_efficientnet_b4(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.efficientnet_b4(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


def build_model_efficientnet_b7(num_classes: int, pretrained: bool = False):
    kwargs = _torchvision_pretrained_kwargs(pretrained)
    model = tv_models.efficientnet_b7(**kwargs)
    return _prepare_torchvision_model(model, num_classes)


# timm-based builders (vision transformers, convnext, swin, etc.)
def _make_timm_builder(timm_name: str) -> Callable[[int, bool], Tuple[nn.Module, int]]:
    def _builder(num_classes: int, pretrained: bool = False):
        return _build_timm_model(timm_name, num_classes, pretrained)
    return _builder


# register timm names used in your original code
build_model_vit_base = _make_timm_builder("vit_base_patch16_224")
build_model_vit_large = _make_timm_builder("vit_large_patch16_224")
build_model_vit_huge = _make_timm_builder("vit_huge_patch14_224")
build_model_deit_base = _make_timm_builder("deit_base_patch16_224")
build_model_deit_small = _make_timm_builder("deit_small_patch16_224")
build_model_deit_tiny = _make_timm_builder("deit_tiny_patch16_224")
build_model_convnext_tiny = _make_timm_builder("convnext_tiny")
build_model_convnext_small = _make_timm_builder("convnext_small")
build_model_convnext_base = _make_timm_builder("convnext_base")
build_model_convnext_large = _make_timm_builder("convnext_large")
build_model_convnext_xlarge = _make_timm_builder("convnext_xlarge")
build_model_swin_tiny = _make_timm_builder("swin_tiny_patch4_window7_224")
build_model_swin_small = _make_timm_builder("swin_small_patch4_window7_224")
build_model_swin_base = _make_timm_builder("swin_base_patch4_window7_224")
build_model_swin_large = _make_timm_builder("swin_large_patch4_window7_224")
build_model_swin_v2_tiny = _make_timm_builder("swinv2_tiny_window8_256")
build_model_swin_v2_small = _make_timm_builder("swinv2_small_window8_256")
build_model_swin_v2_base = _make_timm_builder("swinv2_base_window8_256")
build_model_swin_v2_large = _make_timm_builder("swinv2_large_window12_192_22k")
build_model_efficientnetv2_s = _make_timm_builder("efficientnetv2_s")
build_model_efficientnetv2_m = _make_timm_builder("efficientnetv2_m")
build_model_efficientnetv2_l = _make_timm_builder("efficientnetv2_l")

# ----------------------------
# Registry
# ----------------------------
def get_backbone_builder(name: str):
    name = name.lower()
    mapping = {
        "resnet18": build_model_resnet18,
        "resnet50": build_model_resnet50,
        "mobilenet_v2": build_model_mobilenet_v2,
        "mobilenet_v3_small": build_model_mobilenet_v3_small,
        "mobilenet_v3_large": build_model_mobilenet_v3_large,
        "efficientnet_b0": build_model_efficientnet_b0,
        "efficientnet_b3": build_model_efficientnet_b3,
        "efficientnet_b4": build_model_efficientnet_b4,
        "efficientnet_b7": build_model_efficientnet_b7,
        "vit_base": build_model_vit_base,
        "vit_large": build_model_vit_large,
        "vit_huge": build_model_vit_huge,
        "deit_base": build_model_deit_base,
        "deit_small": build_model_deit_small,
        "deit_tiny": build_model_deit_tiny,
        "convnext_tiny": build_model_convnext_tiny,
        "convnext_small": build_model_convnext_small,
        "convnext_base": build_model_convnext_base,
        "convnext_large": build_model_convnext_large,
        "convnext_xlarge": build_model_convnext_xlarge,
        "swin_tiny": build_model_swin_tiny,
        "swin_small": build_model_swin_small,
        "swin_base": build_model_swin_base,
        "swin_large": build_model_swin_large,
        "swin_v2_tiny": build_model_swin_v2_tiny,
        "swin_v2_small": build_model_swin_v2_small,
        "swin_v2_base": build_model_swin_v2_base,
        "swin_v2_large": build_model_swin_v2_large,
        "efficientnetv2_s": build_model_efficientnetv2_s,
        "efficientnetv2_m": build_model_efficientnetv2_m,
        "efficientnetv2_l": build_model_efficientnetv2_l,
    }
    if name not in mapping:
        raise ValueError(f"Unknown backbone model name: {name}")
    return mapping[name]
