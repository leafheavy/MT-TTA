from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from corruption_method import *

set_seed(seed)

def compute_fisher_matrix(mymodel, fisher_loader, device='cpu'):
    """
    计算Fisher矩阵
    - args:
        - mymodel: 训练好的模型
        - fisher_loader: 用于计算 Fisher 矩阵的数据加载器 (源域)
    - returns:
        - fishers: 针对源域计算得到的 Fisher 矩阵
    """
    # 初始化Fisher矩阵
    fishers = {}
    mymodel.train()  # 确保模型处于训练模式
    for iter_, (inputs, labels) in tqdm(enumerate(fisher_loader, start=1), desc="Computing Fisher Matrix", total=len(fisher_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播 --> 获得伪标签
        outputs = mymodel(inputs) # list of tensor, i.e., [task1_out, task2_out, ...]
        
        # 计算总损失（假设多任务损失之和）
        total_loss = 0
        for i, out in enumerate(outputs):
            # 得到硬标签, shape (batch_size,)
            _, targets = out.max(1)
            targets = targets.detach() # 从计算图中分离 (detach)，使其成为一个固定目标

            # supervised learning
            # loss_i = F.cross_entropy(out, labels[:, i])

            # unsupervised learning
            loss_i = F.cross_entropy(out, targets, reduction='none') # shape (batch_size,)

            loss_i_avg = loss_i.mean() # 转换为标量
            total_loss += loss_i_avg
            # task_losses.append(loss_i_avg)
        # task_losses = torch.stack(task_losses).unsqueeze(1)  # shape (num_tasks, 1)
        
        # 反向传播计算梯度
        total_loss.backward()
        
        # 更新Fisher矩阵
        for name, param in mymodel.named_parameters():
            if 'bn' in name.lower() or isinstance(param, nn.BatchNorm2d): # 只对 BN 层的参数计算 Fisher 矩阵
                if param.grad is not None:
                    if iter_ > 1 and name in fishers:
                        fisher = param.grad.data ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data ** 2
                    # 最后一批次取平均
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers[name] = [fisher.detach(), param.data.detach().clone()]
        
        # 清除梯度
        mymodel.zero_grad()

    return fishers
