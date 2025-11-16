from copy import deepcopy
from sklearn.metrics import f1_score
import argparse
import numpy as np
import torch
from torch import nn

# Hyper-parameters Controller
def parse_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters Controller")
    
    # General hyper-parameters    
    parser.add_argument('--lr_tta', type=float, default=2.5e-3, help='The learning rate during adaptation')
    parser.add_argument('--image_size', type=int, default=224, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tta_steps', type=int, default=10, help='The adaptation steps')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--random_seed', type=int, default=42)
    
    # CoCo specific hyper-parameters
    parser.add_argument('--GC_c', type=float, default=0.5, help='The search range of Gradient Consensus')
    parser.add_argument('--PC_alpha', type=float, default=2000.0, help='The strength of Plasticity Constraints')

    args = parser.parse_args()
    return args

plantdata_task_classes = [8, 5, 19] # PlantData: Number of classes per task
celeba_task_classes = [2, 2, 2, 2]  # CelebA: Number of classes per task (4 tasks: 'Attractive', 'Male', 'Smiling', 'Wearing_Lipstick')
datastes_path = r'Datasets/'  # Dataset path
filedir = 'pretrained_models/' # Pre-trained model save path

# Evaluating
def evaluate_model(model, dataloader, task_names, loss_func=None, require_loss=False, device='cpu'):
    """
    - args:
        - model
        - dataloader
        - task_names: for mapping with task-wise f1 scores
        - loss_func: function to calculate loss
        - require_loss: whether to return loss values
        - device
    - return:
        - weighted f1 score
        - avg_task_losses: shape (num_task, ) (optional)
        - total_loss: scaler (optional)
    """

    model.eval()
    epoch_task_losses = []  # Loss values for each task in the current epoch

    task_preds = {task: [] for task in task_names}
    task_labels = {task: [] for task in task_names}

    supervised_flag = False

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss(reduction="none")
        supervised_flag = True

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)
            outputs = model(inputs)

            # Calculate loss for each task
            task_losses_per_batch = []
            for idx, out in enumerate(outputs):
                if supervised_flag:
                    loss_i = loss_func(out, labels[:, idx])  # shape (batch_size,) supervised learning
                else:
                    loss_i = loss_func(out)                  # shape (batch_size,) unsupervised learning
                
                loss_i_avg = loss_i.mean()  # Convert to scalar
                task_losses_per_batch.append(loss_i_avg)

                probs = torch.softmax(out, dim=1)
                _, preds = torch.max(probs, dim=1)
                task_name = task_names[idx]
                task_preds[task_name].append(preds.cpu().numpy())
                task_labels[task_name].append(labels[:, idx].cpu().numpy())
            
            task_losses_per_batch = torch.stack(task_losses_per_batch)  # shape (num_tasks, 1)
            epoch_task_losses.append(task_losses_per_batch.detach().cpu().numpy())

        # Calculate average loss across batches
        avg_task_losses = np.mean(epoch_task_losses, axis=0)
        total_loss = avg_task_losses.mean()

    # Calculate F1-score for each task
    results = {}
    for task in task_names:
        # print(f"Task {task}: collected {len(task_labels[task])} batches of predictions")
        
        # Check if empty
        if len(task_labels[task]) == 0:
            print(f"Warning: No predictions collected for task {task}")
            results[task] = 0.0
            continue
            
        try:
            y_true = np.concatenate(task_labels[task])
            y_pred = np.concatenate(task_preds[task])
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            results[task] = f1
        except Exception as e:
            print(f"Error calculating F1 for task {task}: {e}")
            results[task] = 0.0

    if require_loss:
        return results, avg_task_losses, total_loss
    else:
        return results

# Configure model BN/IN layer status --> for TTA methods using ResNet-GN/Vit-LN models
def configure_model(model):
    model.train()
    model.requires_grad_(False)
    
    # Enable grad + force batch statistics
    for nm, m in model.named_modules():
        # BatchNorm layers
        if 'bn' in nm.lower() or isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)

            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        elif 'ln' in nm.lower() or 'gn' in nm.lower() or isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    
    return model

# Check for intersection between train_loader and test_loader
def check_dataloader_intersection(train_loader, test_loader):
    # Get indices from training set
    train_indices = set()
    if hasattr(train_loader.dataset, 'indices'):
        train_indices = set(train_loader.dataset.indices) # If it's a Subset
    else:
        train_indices = set(range(len(train_loader.dataset))) # If it's a full dataset
    
    # Get indices from test set
    test_indices = set()
    if hasattr(test_loader.dataset, 'indices'):
        test_indices = set(test_loader.dataset.indices) # If it's a Subset
    else:
        test_indices = set(range(len(test_loader.dataset))) # If it's a full dataset
    
    # Calculate intersection
    intersection = train_indices & test_indices
    
    print(f"Training set sample count: {len(train_indices)}")
    print(f"Test set sample count: {len(test_indices)}")
    print(f"Intersection sample count: {len(intersection)}")
    
    if len(intersection) > 0:
        print("[WARNING] Training set and test set have intersection!")
        print(f"Intersection index examples: {list(intersection)[:10]}")
        return 
    else:
        print("Training set and test set have no intersection, data split is correct.")
        return

# Controling randomness
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
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