from GradientConsensus import compute_gradient, gradient_consensus
from utils import *

class CoCo(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, tta_params=None, grad_adjust_fn=gradient_consensus, require_fishers=False, fishers=None, fisher_alpha=1.0, specific_task=None):
        """
        - args:
            - model: Model to adapt
            - steps: 0/1 --> inference/adaption
            - episodic: True/False --> Online/Offline
            - tta_params: Learnable parameter
            - grad_adjust_fn: Multi-task adaptation algorithm
            - require_fishers: Enable Plasticity Constraints (PC)
            - fishers: Fishers Information Dict
            - fisher_alpha: PC strength
            - specific_task: Use gradient of specific task to update learnable parameter
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.grad_adjust_fn = grad_adjust_fn
        self.require_fishers = require_fishers
        self.fishers = fishers  # dict: {param_name: (F_matrix, theta_star)}
        self.fisher_alpha = fisher_alpha
        self.specific_task = specific_task
        self.multitask_loss = []

        # Collect parameters to adapt (default collect weight/bias from BN)
        if tta_params is None:
            params_list, names = collect_params(self.model)
            self.tta_params = params_list
            self.tta_param_names = names
        else:
            self.tta_params = tta_params
            self.tta_param_names = None

        # Save initial state (for episodic reset)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        if self.steps > 0:
            # Perform forward + PC + GC + step
            for _ in range(self.steps):
                outputs = self.forward_and_adapt_coco(x)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def forward_and_adapt_coco(self, x):
        with torch.enable_grad():
            outputs = self.model(x)

            # Calculate entropy loss for each task (scalar)
            task_losses = []
            for out in outputs:
                loss_i = softmax_entropy(out)  # shape: (batch_size,)
                task_losses.append(loss_i.mean())
            task_losses = torch.stack(task_losses)  # (num_tasks,)

            # Total loss
            if self.require_fishers and (self.fishers is not None):
                ewc_loss = sum(self.fisher_alpha * (self.fishers[name][0] * (p - self.fishers[name][1])**2).sum()
                               for name, p in self.model.named_parameters() if name in self.fishers)
                
                # 2 method to achieve EWC loss and entropy loss have different scales
                # (1) Distribute EWC loss equally
                # num_tasks = len(task_losses)
                # ewc_value = (ewc_loss / float(num_tasks)).to(task_losses.device)
                # ewc_loss_per_task = ewc_value.expand(num_tasks)   # shape: scaler --> 1D tensor (num_tasks,)
                
                # (2) Copy EWC loss num_tasks times
                num_tasks = len(task_losses)
                ewc_value = ewc_loss.to(task_losses.device)
                ewc_loss_per_task = ewc_value.expand(num_tasks)   # shape: scaler --> 1D tensor (num_tasks,)

                task_losses += ewc_loss_per_task

            self.optimizer.zero_grad()

            regularized_grad, weights = compute_gradient(
                task_losses.unsqueeze(1),            # (num_tasks, 1)
                self.tta_params,
                grad_adjust_fn=self.grad_adjust_fn,
                allow_unused=False,
                retain_graph=False,
                specific_task=self.specific_task
            )

            # 最后一步更新适应参数
            self.optimizer.step()

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)