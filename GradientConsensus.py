"""
Code for searching consensusal gradient for model adaptation, based on CAGrad.
GitHub Address: https://github.com/Cranial-XIX/CAGrad

Paper: Conflict-Averse Gradient Descent for Multi-task Learning
@article{liu2021conflict,
  title={Conflict-Averse Gradient Descent for Multi-task Learning},
  author={Liu, Bo and Liu, Xingchao and Jin, Xiaojie and Stone, Peter and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
"""

from typing import Iterable, List, Optional
import torch
from torch import TensorType

from utils import *

set_seed(seed)

def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device

def apply_vector_grad_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor], accumulate: bool = False):
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")

    pointer = 0
    # 一律在 no_grad() 下修改 param.grad，避免被 autograd 跟踪
    with torch.no_grad():
        for param in parameters:
            num_param = param.numel()
            # 切片并reshape为 param 的形状，确保dtype/device与 param 匹配
            grad_slice = vec[pointer:pointer + num_param].view_as(param)
            grad_tensor = grad_slice.detach().clone().to(param.device, dtype=param.dtype)

            if accumulate:
                # print("Using gradient accumulation mode.")
                # 与 autograd 行为一致：若 param.grad is None，则直接赋值为当前梯度（而不是先创建 0 再累加）
                if param.grad is None:
                    param.grad = grad_tensor
                else:
                    # 原地累加，语义与 backward 保持一致（也减少额外拷贝）
                    param.grad.add_(grad_tensor)
            else:
                # print("Using gradient overwrite mode.")
                # 覆盖行为：直接将 grad 赋给 param.grad（与 backward 在第一次创建时直接赋值的语义一致）
                param.grad = grad_tensor

            pointer += num_param

def gradient_consensus(
    grad_vec: torch.Tensor,
    num_tasks: int,
    GC_c: float,
    iters: int = 100,
    lr_default: float = 1.0,
    eps: float = 1e-4,
    device = 'cpu'
):
    """
    grad_vec: (num_tasks, dim)
    returns: regularized gradient (dim,), and final/selected ww (num_tasks,)
    """
    grads = grad_vec  # (num_tasks, num_params)

    # 计算协方差矩阵并缩放
    GG = grads.mm(grads.t()).to('cpu')  # (num_tasks, num_tasks)
    scale = (torch.diag(GG) + eps).sqrt().mean()  # 得到 L2 范数的均值
    GG = GG / scale.pow(2)  # 进行 L2 Normalization

    GG_dev = GG.to(device)
    Gg = GG_dev.mean(1, keepdims=True)  # 每个任务与其他任务梯度的平均相关性, shape: (num_tasks, 1)
    gg = Gg.mean(0, keepdims=True)      # gradient的均值 (i.e., g_0)

    w = torch.zeros(num_tasks, 1, requires_grad=True, device=device)
    # w_opt = torch.optim.SGD([w], lr=lr_default, momentum=0.5)
    w_opt = torch.optim.Adam([w], lr=lr_default, betas=(0.9, 0.999), weight_decay=1e-4)

    phi_sqrt = (gg + eps).sqrt() * GC_c  # phi.sqrt()

    w_best = None
    obj_best = float('inf')

    # entropy_weight = 0.01
    for i in range(iters):
        w_opt.zero_grad()
        ww = torch.softmax(w, 0) # 将权重转换为概率分布, 保证 w 的和为 1 (num_tasks,1)
        # 计算目标函数
        # Term 1: g_w.T * g_0 = \sum_i w_i (g_0 * g_i) 
        #                     = \sum_i w_i <g_0, g_i> 
        #                     = ww.t().mm(Gg)
        term1 = ww.t().mm(Gg)  # (1,1)
        # Term 2: phi.sqrt() * || g_w || = (g_w.T * g_w).sqrt() 
        #                                = (\sum_i w_i^2 * (g_i * g_i)).sqrt()
        #                                = (\sum_i w_i^2 * <g_i, g_i>).sqrt() 
        #                                = (ww.t().mm(GG).mm(ww)).sqrt()
        term2 = (ww.t().mm(GG_dev).mm(ww) + eps).sqrt()  # (1,1)

        obj = term1 + phi_sqrt * term2

        # # 熵正则化项
        # term3 = -torch.sum(ww * torch.log(ww + 1e-12)) * entropy_weight
        # obj = term1 + phi_sqrt * term2 - term3
        
        # 优化问题中止条件
        obj_scalar = obj.item()
        # track best
        if obj_scalar < obj_best:
            obj_best = obj_scalar
            # clone to keep best parameters (detach to avoid referencing graph)
            w_best = w.clone().detach()
        if i < iters - 1:
            obj.backward()
            w_opt.step()

    # 计算最终梯度
    ww = torch.softmax(w_best, 0)  # (num_tasks,1)
    gw_norm = (ww.t().mm(GG_dev).mm(ww) + eps).sqrt()

    lambda_frac = phi_sqrt.view(-1) / (gw_norm + eps) # 得到 frac{1}{\lambda}

    grads_dev = grads.to(device) # ensure grads is on same device

    """
    d^* =          g_0                     +  \frac{1}{\lambda}     *                 g_{w_{(m)}}
        = 1/M \sum_{m=1}^{M} * g_{(m)}     +  \frac{1}{\lambda})    *  \sum_{m=1}^{M} w_{(m)} * g_{(m)})
        =    \sum_{m=1}^{M}  * [1/M        +  \frac{1}{\lambda}     *               (w_{(m)}            ] * g_{(m)} 

    """
    # g = ((1/num_tasks + ww * lambda_frac).view(-1, 1).to(grads_dev.device) * grads_dev).sum(0) / (1 + GC_c**2) # 即 d^*
    g = ((1/num_tasks + ww * lambda_frac).view(-1, 1).to(grads_dev.device) * grads_dev).sum(0) # without scaled

    return g, ww.view(-1)

def compute_gradient(
    task_loss: torch.Tensor,
    parameters: List[torch.Tensor],
    grad_adjust_fn = None,
    GC_c = 0.5,
    allow_unused: bool = False,
    retain_graph: bool = False,
    specific_task: int = None,
    accumulate: bool = False,
    device = 'cpu'
):
    """
    - args:
        - task_loss.shape: (num_tasks, 1)-> Torch.Tensor
        - parameters: 模型参数
        - grad_adjust: Gradient Consensus 的方法, None 则为简单平均
        - specific_task: 只取特定任务的梯度
    - return:
        - grad_norms: 记录每个任务的梯度范数, 用以检查梯度范数是否根据不同的梯度调整方法而变化
        - weight_logs: 记录每个任务的权重
    """
    
    if specific_task is not None:
        grads_tuple = torch.autograd.grad(
            task_loss[specific_task],
            parameters,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )
        grads_clean = [
            g.contiguous() if g is not None else torch.zeros_like(p)
            for p, g in zip(parameters, grads_tuple)
        ]
        regularized_grad = torch.nn.utils.parameters_to_vector(grads_clean).to(device)
        weights = None
    else:
        num_tasks = int(task_loss.shape[0])
        grads = []
        # 计算每个任务的梯度
        for idx in range(num_tasks):
            single_loss = task_loss[idx]
            # keep graph for all except last unless retain_graph True
            rg = retain_graph or (idx != num_tasks - 1)
            grads_tuple = torch.autograd.grad(single_loss, parameters, retain_graph=rg, allow_unused=allow_unused)
            # convert None grads -> zeros of same shape & dtype & device
            grads_clean = []
            for p, g in zip(parameters, grads_tuple):
                if g is None:
                    grads_clean.append(torch.zeros_like(p))
                else:
                    grads_clean.append(g.contiguous())
            grads.append(tuple(grads_clean))

        # flatten to matrix (num_tasks, dim)
        grad_vec = torch.cat(
            [torch.nn.utils.parameters_to_vector(g).unsqueeze(0) for g in grads], dim=0
        ).to(device)

        # Simple mean method (M-TENT)
        if grad_adjust_fn is None:
            regularized_grad = grad_vec.mean(dim=0)
            weights = None
        else:
            regularized_grad, weights = grad_adjust_fn(grad_vec, num_tasks, GC_c)
            # ensure returned grad on same device
            regularized_grad = regularized_grad.to(device)
            if weights is not None:
                weights = weights.to('cpu').detach().numpy()

    # apply vector gradient back to parameters
    apply_vector_grad_to_parameters(regularized_grad, parameters, accumulate=accumulate)

    return regularized_grad, weights