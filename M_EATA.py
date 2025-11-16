"""
Code for EATA's MultiTask version, based on EATA.
GitHub Address: https://github.com/mr-eggplant/EATA

Copyright to EATA ICML 2022 Authors, 2022.03.20
Paper: Efficient Test-Time Model Adaptation without Forgetting
@InProceedings{niu2022efficient,
  title={Efficient Test-Time Model Adaptation without Forgetting},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Chen, Yaofo and Zheng, Shijian and Zhao, Peilin and Tan, Mingkui},
  booktitle = {The Internetional Conference on Machine Learning},
  year = {2022}
}
"""
import math
import torch.nn.functional as F

from utils import *

# multi-task version of EATA
class EATA_MultiTask(nn.Module):
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)*0.40, d_margin=0.05, current_model_probs=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.current_model_probs = current_model_probs
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata_mutilltask(
                    x, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs,
                    fisher_alpha=self.fisher_alpha, d_margin=self.d_margin, num_samples_update=self.num_samples_update_2
                )
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # pass

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

@torch.enable_grad()
def forward_and_adapt_eata_mutilltask(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=2000.0, d_margin=0.05, num_samples_update=0):

    outputs = model(x)

    task_entropies = [softmax_entropy(out) for out in outputs] # 计算每个任务的熵
    entropys = torch.mean(torch.stack(task_entropies), dim=0)  # 多任务熵取平均

    filter_ids_1 = torch.where(entropys < e_margin)
    
    # 对每个任务的输出应用第一次过滤
    filtered_outputs = [out[filter_ids_1] for out in outputs]
    
    # 第二次过滤（冗余样本）
    if current_model_probs is not None and len(filtered_outputs[0]) > 0:
        
        # task_probs = filtered_outputs[0].softmax(1) # 使用第一个任务计算余弦相似度
        
        # 按任务求熵，再取平均
        task_probs = [out.softmax(1) for out in filtered_outputs]
        avg_entropy = torch.mean(torch.stack([softmax_entropy(p) for p in task_probs]), dim=0)

        task_probs = filtered_outputs[0].softmax(1)
        cosine_sim = F.cosine_similarity(current_model_probs.unsqueeze(0), task_probs, dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_sim) < d_margin)
        
        # 应用第二次过滤
        final_outputs = [out[filter_ids_2] for out in filtered_outputs]
        updated_probs = update_model_probs(current_model_probs, final_outputs[0].softmax(1))
    else:
        final_outputs = filtered_outputs
        updated_probs = update_model_probs(current_model_probs, filtered_outputs[0].softmax(1) if len(filtered_outputs[0]) > 0 else None)
    
    optimizer.zero_grad()

    # 计算多任务损失
    loss = 0
    valid_samples = len(final_outputs[0])
    # print(f"Valid samples after filtering: {valid_samples}")
    if valid_samples > 0:
        for out, original_out in zip(final_outputs, outputs):
            task_loss = softmax_entropy(out).mean()
            loss += task_loss
        
        # 添加Fisher正则项
        if fishers is not None:
            ewc_loss = sum(fisher_alpha * (fishers[name][0] * (p - fishers[name][1])**2).sum() 
                          for name, p in model.named_parameters() if name in fishers)
            loss += ewc_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
    
    return outputs, valid_samples, len(filter_ids_1[0]), updated_probs