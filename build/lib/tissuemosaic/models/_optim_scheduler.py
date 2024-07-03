from typing import Tuple
import math
import torch
from torch.optim.optimizer import Optimizer


def linear_warmup_and_cosine_protocol(
        f_values: Tuple[float, float, float],
        x_milestones: Tuple[int, int, int, int]):
    """
    There are 5 regions:
    1. constant at f0 for x < x0
    2. linear increase from f0 to f1 for x0 < x < x1
    3. constant at f1 for x1 < x < x2
    4. cosine protocol from f1 to f2 for x2 < x < x3
    5. constant at f2 for x > x3

    If you want a linear_ramp followed by a cosine_decay only simply set:
    1. x0=0 (to eliminate the first constant piece)
    2. x2=x1 (to eliminate the second constant piece)
    3. max_epochs=x3 (to make the simulation stop after the linear or cosine decay)
    """
    assert x_milestones[0] <= x_milestones[1] <= x_milestones[2] <= x_milestones[3]

    def fn(step):
        if step <= x_milestones[0]:
            return float(f_values[0])
        elif (step > x_milestones[0]) and (step <= x_milestones[1]):
            m = float(f_values[1] - f_values[0]) / float(max(1, x_milestones[1] - x_milestones[0]))
            return float(f_values[0]) + m * float(step - x_milestones[0])
        elif (step > x_milestones[1]) and (step <= x_milestones[2]):
            return float(f_values[1])
        elif (step > x_milestones[2]) and (step <= x_milestones[3]):
            progress = float(step - x_milestones[2]) / float(max(1, x_milestones[3] - x_milestones[2]))  # in (0,1)
            tmp = 0.5 * (1.0 + math.cos(math.pi * progress))  # in (1,0)
            return float(f_values[2]) + tmp * float(f_values[1] - f_values[2])
        else:
            return float(f_values[2])

    return fn


class LARS(Optimizer):
    """
    Extends SGD in PyTorch with LARS scaling from the paper
    'Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>'_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
        eps (float, optional): eps for division denominator (default: 1e-8)

    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> input = torch.Tensor(10)
        >>> target = torch.Tensor([1.])
        >>> loss_fn = lambda input, target: (input - target) ** 2
        >>> #
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        The application of momentum in the SGD part is modified according to
        the PyTorch standards. LARS scaling fits into the equation in the
        following fashion.
        .. math::
            \begin{aligned}
                g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
                v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \\end{aligned}
        where :math:`p`, :math:`g`, :math:`v`, :math:`\\mu` and :math:`\beta` denote the
        parameters, gradient, velocity, momentum, and weight decay respectively.
        The :math:`lars_lr` is defined by Eq. 6 in the paper.
        The Nesterov version is analogously modified.
    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded from
        layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
        and BYOL.
    """

    def __init__(
        self,
        params,
        lr=None,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-8,
    ):
        if lr is None or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["trust_coefficient"]

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
