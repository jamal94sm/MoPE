"""
tent.py — TENT: Fully Test-Time Adaptation by Entropy Minimization.

Wang et al., ICLR 2021.
Aligned with: https://github.com/DequanWang/tent

Core idea: At test time, update only the affine parameters (weight, bias)
of BatchNorm layers by minimizing the entropy of model predictions.
BN running statistics are disabled — only batch statistics are used.

Usage:
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    tented_model = tent.Tent(model, optimizer)
    outputs = tented_model(inputs)  # infers and adapts
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Save initial state for episodic reset
        # Note: if never reset (continual adaptation), this copy is unused
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: original TENT only collects BatchNorm2d. We also include
    BatchNorm1d for models like iResNet100 that use BN1d in the
    final embedding layer.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent.

    Following the original TENT implementation:
    1. Set entire model to train mode (needed for BN to use batch stats)
    2. Disable gradients for all parameters
    3. Re-enable gradients for BN affine parameters only
    4. Disable running stats tracking — force batch statistics

    Note: original only handles BatchNorm2d. We also handle BatchNorm1d
    for iResNet100 compatibility.
    """
    # Train mode — required so BN uses batch statistics
    model.train()

    # Disable all gradients
    model.requires_grad_(False)

    # Enable BN affine params + force batch statistics
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            # Force use of batch stats in both train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


def check_model(model):
    """Check model for compatibility with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"

    param_grads = [p.requires_grad for p in model.parameters()]
    has_any = any(param_grads)
    has_all = all(param_grads)
    assert has_any, "tent needs params to update: check which require grad"
    assert not has_all, "tent should not update all params: check which require grad"

    has_bn = any(isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
                 for m in model.modules())
    assert has_bn, "tent needs normalization for its optimization"


# ══════════════════════════════════════════════════════════════
#  BNA: Batch Norm Adaptation (no gradients, no head)
# ══════════════════════════════════════════════════════════════

def configure_model_bna(model):
    """
    Configure model for BNA (Batch Norm Adaptation).

    1. Set model to eval mode
    2. Reset BN running stats
    3. Set BN to train mode (use batch stats + update running stats)
    4. Freeze all parameters (no gradients at all)
    """
    model.eval()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.train()  # use batch stats + accumulate into running stats

    return model


@torch.no_grad()
def forward_bna(x, model):
    """Forward pass for BNA. No gradients, just updates BN running stats."""
    return model(x)
