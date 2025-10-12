import torch.nn as nn
import torchvision.models as models
import timm

def build_model_resnet18(num_classes: int, pretrained: bool = False):
    model = models.resnet18(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def build_model_resnet50(num_classes: int, pretrained: bool = False):
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def build_model_mobilenet_v2(num_classes: int, pretrained: bool = False):
    model = models.mobilenet_v2(pretrained=pretrained)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def build_model_mobilenet_v3_small(num_classes: int, pretrained: bool = False):
    model = models.mobilenet_v3_small(pretrained=pretrained)
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, num_classes)
    return model

def build_model_mobilenet_v3_large(num_classes: int, pretrained: bool = False):
    model = models.mobilenet_v3_large(pretrained=pretrained)
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, num_classes)
    return model

def build_model_efficientnet_b0(num_classes: int, pretrained: bool = False):
    model = models.efficientnet_b0(pretrained=pretrained)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def build_model_efficientnet_b3(num_classes: int, pretrained: bool = False):
    model = models.efficientnet_b3(pretrained=pretrained)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def build_model_efficientnet_b4(num_classes: int, pretrained: bool = False):
    model = models.efficientnet_b4(pretrained=pretrained)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def build_model_efficientnet_b7(num_classes: int, pretrained: bool = False):
    model = models.efficientnet_b7(pretrained=pretrained)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def build_model_vit_base(num_classes: int, pretrained: bool = False):
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_vit_large(num_classes: int, pretrained: bool = False):
    model = timm.create_model("vit_large_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_vit_huge(num_classes: int, pretrained: bool = False):
    model = timm.create_model("vit_huge_patch14_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_deit_base(num_classes: int, pretrained: bool = False):
    model = timm.create_model("deit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_deit_small(num_classes: int, pretrained: bool = False):
    model = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_deit_tiny(num_classes: int, pretrained: bool = False):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_convnext_tiny(num_classes: int, pretrained: bool = False):
    model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_convnext_small(num_classes: int, pretrained: bool = False):
    model = timm.create_model("convnext_small", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_convnext_base(num_classes: int, pretrained: bool = False):
    model = timm.create_model("convnext_base", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_convnext_large(num_classes: int, pretrained: bool = False):
    model = timm.create_model("convnext_large", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_convnext_xlarge(num_classes: int, pretrained: bool = False):
    model = timm.create_model("convnext_xlarge", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_tiny(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_small(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swin_small_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_base(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_large(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swin_large_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_v2_tiny(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swinv2_tiny_window8_256", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_v2_small(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swinv2_small_window8_256", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_v2_base(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swinv2_base_window8_256", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_swin_v2_large(num_classes: int, pretrained: bool = False):
    model = timm.create_model("swinv2_large_window12_192_22k", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_efficientnetv2_s(num_classes: int, pretrained: bool = False):
    model = timm.create_model("efficientnetv2_s", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_efficientnetv2_m(num_classes: int, pretrained: bool = False):
    model = timm.create_model("efficientnetv2_m", pretrained=pretrained, num_classes=num_classes)
    return model

def build_model_efficientnetv2_l(num_classes: int, pretrained: bool = False):
    model = timm.create_model("efficientnetv2_l", pretrained=pretrained, num_classes=num_classes)
    return model


def get_backbone_builder(name: str):
    name = name.lower()
    if name == "resnet18":
        return build_model_resnet18
    elif name == "resnet50":
        return build_model_resnet50
    elif name == "mobilenet_v2":
        return build_model_mobilenet_v2
    elif name == "mobilenet_v3_small":
        return build_model_mobilenet_v3_small
    elif name == "mobilenet_v3_large":
        return build_model_mobilenet_v3_large
    elif name == "efficientnet_b0":
        return build_model_efficientnet_b0
    elif name == "efficientnet_b3":
        return build_model_efficientnet_b3
    elif name == "efficientnet_b4":
        return build_model_efficientnet_b4
    elif name == "efficientnet_b7":
        return build_model_efficientnet_b7
    elif name == "vit_base":
        return build_model_vit_base
    elif name == "vit_large":
        return build_model_vit_large
    elif name == "vit_huge":
        return build_model_vit_huge
    elif name == "deit_base":
        return build_model_deit_base
    elif name == "deit_small":
        return build_model_deit_small
    elif name == "deit_tiny":
        return build_model_deit_tiny
    elif name == "convnext_tiny":
        return build_model_convnext_tiny
    elif name == "convnext_small":
        return build_model_convnext_small
    elif name == "convnext_base":
        return build_model_convnext_base
    elif name == "convnext_large":
        return build_model_convnext_large
    elif name == "convnext_xlarge":
        return build_model_convnext_xlarge
    elif name == "swin_tiny":
        return build_model_swin_tiny
    elif name == "swin_small":
        return build_model_swin_small
    elif name == "swin_base":
        return build_model_swin_base
    elif name == "swin_large":
        return build_model_swin_large
    elif name == "swin_v2_tiny":
        return build_model_swin_v2_tiny
    elif name == "swin_v2_small":
        return build_model_swin_v2_small
    elif name == "swin_v2_base":
        return build_model_swin_v2_base
    elif name == "swin_v2_large":
        return build_model_swin_v2_large
    elif name == "efficientnetv2_s":
        return build_model_efficientnetv2_s
    elif name == "efficientnetv2_m":
        return build_model_efficientnetv2_m
    elif name == "efficientnetv2_l":
        return build_model_efficientnetv2_l
    else:
        raise ValueError(f"Unknown backbone model name: {name}")
