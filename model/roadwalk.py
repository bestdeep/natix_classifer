# models/custom_model.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

# ---------------------------
# Gradient Reversal Layer (for DANN)
# ---------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        return grad_output.neg() * lambda_, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ---------------------------
# Custom classifier head (flexible)
# ---------------------------
class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 512, dropout: float = 0.4, num_classes: int = 2):
        super().__init__()
        if hidden is None or hidden <= 0:
            # single linear layer
            self.net = nn.Sequential(nn.Linear(in_features, num_classes))
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, num_classes)
            )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Domain classifier head (for adversarial domain adaptation)
# ---------------------------
class DomainHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.4, num_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Main custom model
# ---------------------------
class RoadworkClassifier(nn.Module):
    """
    RoadworkClassifier using timm backbone feature extractor.
    Supports optional domain-adversarial head for sim->real adaptation.

    Args:
      backbone_name: timm model name, e.g. "convnext_base", "swinv2_base_window8_256", "resnet50"
      pretrained: whether to use pretrained weights
      num_classes: final classifier classes (2 for binary)
      head_hidden: hidden dims for classifier head
      domain_adapt: whether to enable domain head (DANN)
      domain_hidden: hidden dim for domain classifier
      grl_lambda: multiplier for GRL (set dynamically during training too)
      dropout: dropout for classifier
    """
    def __init__(
        self,
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        num_classes: int = 2,
        head_hidden: int = 512,
        domain_adapt: bool = True,
        domain_hidden: int = 256,
        grl_lambda: float = 1.0,
        dropout: float = 0.4,
    ):
        super().__init__()
        # Create timm feature extractor: num_classes=0 returns features before final head
        # global_pool='avg' ensures a fixed vector per image
        self.backbone_name = backbone_name
        try:
            self.feature_extractor = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
            )
            # timm models expose num_features
            if not hasattr(self.feature_extractor, "num_features"):
                raise ValueError(f"Backbone {backbone_name} does not expose `num_features`.")
            feat_dim = int(self.feature_extractor.num_features)
        except Exception as e:
            self.feature_extractor_backbone = models.efficientnet_b7(pretrained=pretrained)
            feat_dim = self.feature_extractor_backbone.classifier[1].in_features
            self.feature_extractor_backbone.classifier = torch.nn.Identity()
            self.feature_extractor = self.feature_extractor_backbone
        
        
        self.classifier = ClassifierHead(in_features=feat_dim, hidden=head_hidden, dropout=dropout, num_classes=num_classes)

        # Domain adaptation components
        self.domain_adapt = domain_adapt
        if domain_adapt:
            self.grl = GradientReversal(lambda_=grl_lambda)
            self.domain_head = DomainHead(in_features=feat_dim, hidden=domain_hidden, dropout=dropout, num_domains=2)
        else:
            self.grl = None
            self.domain_head = None

        # small init for heads
        self._init_weights()

    def _init_weights(self):
        # initialize classifier and domain head linears
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.classifier.apply(_init)
        if self.domain_head is not None:
            self.domain_head.apply(_init)

    def freeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    def set_grl_lambda(self, lambda_: float):
        if self.grl is not None:
            self.grl.lambda_ = lambda_

    def forward(self, x, return_features: bool = False, grl_lambda: Optional[float] = None):
        """
        Args:
          x: input image tensor (B,C,H,W)
          return_features: whether to return raw features along with logits
          grl_lambda: optional override for the GRL lambda for this forward pass
        Returns:
          If domain_adapt enabled:
            (class_logits, domain_logits, features)
          else:
            (class_logits, features) if return_features else class_logits
        """
        # features: shape (B, feat_dim)
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)

        if self.domain_adapt:
            if grl_lambda is not None:
                self.set_grl_lambda(grl_lambda)
            # apply GRL then domain head
            rev = self.grl(features)
            domain_logits = self.domain_head(rev)
            if return_features:
                return class_logits, domain_logits, features
            return class_logits, domain_logits

        if return_features:
            return class_logits, features
        return class_logits
