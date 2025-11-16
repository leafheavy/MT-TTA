"""
Code for SAR's MultiTask version, based on EATA.
GitHub Address: https://github.com/mr-eggplant/SAR

Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
Paper: Towards Stable Test-Time Adaptation in Dynamic Wild World
@inproceedings{niu2023towards,
  title={Towards Stable Test-Time Adaptation in Dynamic Wild World},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Wen, Zhiquan and Chen, Yaofo and Zhao, Peilin and Tan, Mingkui},
  booktitle = {Internetional Conference on Learning Representations},
  year = {2023}
}
"""
import math
from typing import List
import torch

from utils import *

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# multi-task version of SAR
class SAR_MultiTask(nn.Module):
    def __init__(self, model, optimizer, task_weights=None, steps=1, episodic=False,
                 margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        
        # 任务权重处理
        num_tasks = len(model.task_heads)
        self.task_weights = task_weights or [1.0/num_tasks] * num_tasks
        
        self.margin_e0 = margin_e0
        self.reset_constant_em = reset_constant_em
        self.ema = None
        
        # # 状态保存
        # self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        if self.steps > 0:
            for _ in range(self.steps):
                step_outputs, ema, reset_flag = forward_and_adapt_sar_mutilltask(
                    x, self.model, self.optimizer,
                    margin=self.margin_e0,
                    reset_constant=self.reset_constant_em,
                    ema=self.ema,
                    task_weights=self.task_weights
                )
                outputs = step_outputs  # 更新outputs
                if reset_flag:
                    self.reset()
                self.ema = ema
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)

        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)
        self.ema = None

@torch.jit.script
def multi_task_entropy(outputs: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    total_entropy = torch.zeros(outputs[0].shape[0], device=outputs[0].device)
    for idx, logits in enumerate(outputs):
        p = logits.softmax(1)
        entropy = -(p * p.log_softmax(1)).sum(1)  # 保持数值稳定性
        total_entropy += weights[idx] * entropy
    return total_entropy

@torch.enable_grad()
def forward_and_adapt_sar_mutilltask(x, model, optimizer, margin, reset_constant, ema, task_weights):
    """适配列表输出的多任务适应"""
    optimizer.zero_grad()
    
    # 第一次前向
    outputs = model(x)
    
    # 计算多任务熵
    # entropys = multi_task_entropy(outputs, task_weights)
    task_entropies = [softmax_entropy(out) for out in outputs] # 计算每个任务的熵
    entropys = torch.mean(torch.stack(task_entropies), dim=0)  # 多任务熵取平均
    
    # 样本过滤
    filter_ids_1 = torch.where(entropys < margin)
    filtered_entropys = entropys[filter_ids_1]
    
    if filtered_entropys.numel() == 0:
        return outputs, ema, False
    
    loss = filtered_entropys.mean()
    loss.backward()
    
    # 第一次优化步骤
    optimizer.first_step(zero_grad=True)
    
    # 第二次前向
    outputs_second = model(x)
    # entropys2 = multi_task_entropy(outputs_second, task_weights)
    task_entropies = [softmax_entropy(out) for out in outputs_second] # 计算每个任务的熵
    entropys2 = torch.mean(torch.stack(task_entropies), dim=0)  # 多任务熵取平均
    
    # 再次过滤
    filtered_entropys2 = entropys2[filter_ids_1]  # 保持第一次的样本索引
    filter_ids_2 = torch.where(filtered_entropys2 < margin)
    final_entropys = filtered_entropys2[filter_ids_2]
    
    # EMA更新
    loss_second = final_entropys.mean() if final_entropys.numel() > 0 else torch.tensor(0.0)
    if not torch.isnan(loss_second):
        ema = update_ema(ema, loss_second.item())
    
    # 第二次反向传播
    if final_entropys.numel() > 0:
        loss_second.backward()
    optimizer.second_step(zero_grad=True)
    
    # 模型恢复判断
    reset_flag = ema is not None and ema < reset_constant
    return outputs, ema, reset_flag