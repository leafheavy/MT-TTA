import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from model import MyModel
from corruption_method import *
from create_data import *
from utils import *

args = parse_args()

# ActMAD-specific augumentation
def argument_data_transforms(image_size=args.image_size, corruption=None, severity=1):
    """
    - args:
        - image_size: Resize 的图像大小
        - corruption: 添加噪声 (可选: gaussian_noise, shot_noise, impulse_noise, defocus_blur, brightness, contrast)
        - severity: 噪声强度 (1-5)
    - return:
        - transforms: 数据增强/Normalization 操作
    """

    transforms_list = [transforms.RandomResizedCrop((image_size, image_size)), transforms.RandomHorizontalFlip()]

    if corruption is not None:
        # 将PIL图像转为numpy数组，应用噪声，再转回Tensor
        transforms_list.extend([
            transforms.Lambda(lambda img: np.array(img)),  # PIL -> numpy (0-255)
            transforms.Lambda(lambda x: corruption(x, severity)),
            transforms.Lambda(lambda x: x.astype(np.float32) / 255.0),
        ])

    # 最终转为Tensor (自动归一化到0-1)
    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)

def Calculating_clean_statsitics(mymodel, dataset, device):
    # 选取要 hook 的层
    chosen_layers = []
    for name, m in mymodel.named_modules():
        if 'bn' in name.lower() or isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            chosen_layers.append(m)
    n_chosen_layers = len(chosen_layers)

    # 创建 hooks（每层一个 SaveEmb）
    hook_list = [SaveEmb() for _ in range(n_chosen_layers)]
    clean_mean, clean_var = [], []

    argument_transforms_clean = argument_data_transforms(corruption=None)
    if dataset == 'plantdata':
        tain_loader_ = create_plantdata(root=datastes_path, split='train', batch_size=args.batch_size, transforms=argument_transforms_clean, proportion=0.7) # PlantData
    elif dataset == 'celeba':
        train_loader_, task_names = create_celeba(root=datastes_path, split='train', batch_size=args.batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'],
                                transforms=argument_transforms_clean, max_samples=None)

    # 遍历训练数据，收集统计量（注意：使用 no_grad）
    for idx, (inputs, _) in tqdm(enumerate(train_loader_), desc='Calculating clean statistics', total=len(train_loader_)):
        # 注册 hooks（使用 hook_list 中的 SaveEmb 实例）
        hooks = [chosen_layers[i].register_forward_hook(hook_list[i]) for i in range(n_chosen_layers)]
        inputs = inputs.to(torch.float32).to(device)
        with torch.no_grad():
            mymodel.eval()
            _ = mymodel(inputs)

            # 对每个 hook 调用 statistics_update() 来把本 batch 的统计加入 int_mean/int_var
            for h in hook_list:
                h.statistics_update()
                h.clear()  # clear outputs 保持内存受控

        # 卸载 hooks
        for hndl in hooks:
            hndl.remove()

        del inputs
        torch.cuda.empty_cache()

    # 现在把每层的训练集统计量保存下来（pop_mean/pop_var）
    for i in range(n_chosen_layers):
        # pop_mean/pop_var 返回的是对所有已 append 的 int_mean/int_var 的平均
        clean_mean.append(hook_list[i].pop_mean().detach())  # detach 成为常数张量
        clean_var.append(hook_list[i].pop_var().detach())

        # 清空 hook_list[i] 的内部累积（以防后续误用）
        hook_list[i].clear()
        hook_list[i].int_mean = []
        hook_list[i].int_var = []

    del train_loader_

    return clean_mean, clean_var, chosen_layers

class SaveEmb:
    def __init__(self):
        self.outputs = []
        self.int_mean = []
        self.int_var = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

    def statistics_update(self):
        self.int_mean.append(torch.mean(torch.vstack(self.outputs), dim=0))
        self.int_var.append(torch.var(torch.vstack(self.outputs), dim=0))

    def pop_mean(self):
        # return torch.mean(torch.stack(self.int_mean), dim=0)

        # 增量计算均值
        if not self.int_mean:
            return None
        
        # 逐个处理，避免大内存分配
        result = self.int_mean[0].clone()
        for i in range(1, len(self.int_mean)):
            result += self.int_mean[i]
        return result / len(self.int_mean)

    def pop_var(self):
        # return torch.mean(torch.stack(self.int_var), dim=0)
        
        # 增量计算方差
        if not self.int_var:
            return None
        
        # 逐个处理，避免大内存分配
        result = self.int_var[0].clone()
        for i in range(1, len(self.int_var)):
            result += self.int_var[i]
        return result / len(self.int_var)

# The pipeline of M-ActMAD
def actmad_pipeline(mymodel, corruption, dataset): # , clean_mean=[], clean_var=[], chosen_layers=[]
    """
    - args
        - mymodel
        - corruption: corruption type
        - dataset
        - clean_mean: Calculating from clean train dataloader
        - clean_var: Calculating from clean train dataloader
        - chosen_layers
    """
    if dataset is None:
        print("Error! Without target dataset!")
        return 
    if corruption is None:
        print("Please specify a corruption method for ActMAD.")
        return
    elif corruption == 'Gaussian Noise':
        argument_transforms_corruption = argument_data_transforms(corruption=gaussian_noise, severity=5)
        transforms_corruption = data_transforms(corruption=gaussian_noise, severity=5)
    elif corruption == 'Shot Noise':
        argument_transforms_corruption = argument_data_transforms(corruption=shot_noise, severity=5)
        transforms_corruption = data_transforms(corruption=shot_noise, severity=5)
    elif corruption == 'Impulse Noise':
        argument_transforms_corruption = argument_data_transforms(corruption=impulse_noise, severity=5)
        transforms_corruption = data_transforms(corruption=impulse_noise, severity=5)
    elif corruption == 'Defocus Blur':
        argument_transforms_corruption = argument_data_transforms(corruption=defocus_blur, severity=5)
        transforms_corruption = data_transforms(corruption=defocus_blur, severity=5)
    elif corruption == 'Brightness':
        argument_transforms_corruption = argument_data_transforms(corruption=brightness, severity=5)
        transforms_corruption = data_transforms(corruption=brightness, severity=5)
    elif corruption == 'Contrast':
        argument_transforms_corruption = argument_data_transforms(corruption=contrast, severity=5)
        transforms_corruption = data_transforms(corruption=contrast, severity=5)

    argument_transforms_clean = argument_data_transforms(corruption=None)
    transforms_clean = data_transforms(corruption=None)

    l1_loss = nn.L1Loss()
    params = list(mymodel.parameters())
    batch_size = 128
    if dataset == 'plantdata':
        lr_tta = 1e-3
        tta_steps = 10
        task_names = plantdata_task_classes_name
    else:
        lr_tta = 0.00025
        tta_steps = 10
        task_names = celeba_task_classes_name
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True) # Following to Code from ActMAD GitHub

    # 选取要 hook 的层
    chosen_layers = []
    for name, m in mymodel.named_modules():
        if 'bn' in name.lower() or isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            chosen_layers.append(m)
    n_chosen_layers = len(chosen_layers)

    # 创建 hooks（每层一个 SaveEmb）
    hook_list = [SaveEmb() for _ in range(n_chosen_layers)]
    clean_mean, clean_var = [], []

    if dataset == 'plantdata':
        train_loader_ = create_plantdata(root=datastes_path, split='train', batch_size=batch_size,
                                        transforms=argument_transforms_clean, proportion=0.7)
    else:
        train_loader_, task_names = create_celeba(root=datastes_path, split='train', batch_size=batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'],
                                     transforms=argument_transforms_clean, max_samples=None) # Only use 134728 (95%) samples because of the limitation of GPU memory

    # 遍历训练数据，收集统计量（注意：使用 no_grad）
    for idx, (inputs, _) in tqdm(enumerate(train_loader_), desc='Calculating clean statistics', total=len(train_loader_)):
        # 注册 hooks（使用 hook_list 中的 SaveEmb 实例）
        hooks = [chosen_layers[i].register_forward_hook(hook_list[i]) for i in range(n_chosen_layers)]
        inputs = inputs.to(torch.float32).to(args.device)
        with torch.no_grad():
            mymodel.eval()
            _ = mymodel(inputs)

            # 对每个 hook 调用 statistics_update() 来把本 batch 的统计加入 int_mean/int_var
            for h in hook_list:
                h.statistics_update()
                h.clear()  # clear outputs 保持内存受控

        # 卸载 hooks
        for hndl in hooks:
            hndl.remove()

        del inputs
        torch.cuda.empty_cache()

    # 现在把每层的训练集统计量保存下来（pop_mean/pop_var）
    for i in range(n_chosen_layers):
        # pop_mean/pop_var 返回的是对所有已 append 的 int_mean/int_var 的平均
        clean_mean.append(hook_list[i].pop_mean().detach())  # detach 成为常数张量
        clean_var.append(hook_list[i].pop_var().detach())

        # 清空 hook_list[i] 的内部累积（以防后续误用）
        hook_list[i].clear()
        hook_list[i].int_mean = []
        hook_list[i].int_var = []

    del train_loader_

    if dataset == 'plantdata':
        te_loader_ = create_plantdata(root=datastes_path, split='test', batch_size=batch_size,
                                      transforms=argument_transforms_corruption, proportion=0.7)
        te_loader = create_plantdata(root=datastes_path, split='test', batch_size=batch_size,
                                     transforms=transforms_corruption, proportion=0.7)
    else:
        te_loader_, _ = create_celeba(root=datastes_path, split='test', batch_size=batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'],
                                   transforms=argument_transforms_corruption, max_samples=None)
        te_loader, _ = create_celeba(root=datastes_path, split='test', batch_size=batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'],
                                  transforms=transforms_corruption, max_samples=None)

    # TTA
    n_chosen_layers = len(chosen_layers)
    for tta_step in tqdm(range(tta_steps), desc="TTA", total=tta_steps):
        for idx, (inputs, labels) in enumerate(te_loader_):
            # 进入训练模式以允许参数更新
            mymodel.train()

            # 如果需要保留 BN 层的 running stats 不被更新，可设置为 evafl
            for name, module in mymodel.named_modules():
                if 'bn' in name.lower() or isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)):
                    module.eval()

            optimizer.zero_grad()

            # 为该 batch 创建新的 SaveEmb hooks（这些实例会收集 module_out 并保持计算图）
            save_outputs_tta = [SaveEmb() for _ in range(n_chosen_layers)]
            hooks_list_tta = [chosen_layers[i].register_forward_hook(save_outputs_tta[i]) for i in range(n_chosen_layers)]

            inputs = inputs.to(args.device)
            # 这里没有 torch.no_grad()，保持计算图
            _ = mymodel(inputs)

            # 对每个 hook 做 statistics_update 并立即 pop 出该 batch 的 mean/var（它们与当前计算图相连）
            act_mean_batch_tta = []
            act_var_batch_tta = []
            for h in save_outputs_tta:
                h.statistics_update()
                # pop_mean/pop_var 返回对 int_mean/int_var 的平均；此时 int_mean 仅包含当前 batch 的一项
                act_mean = h.pop_mean()   # 保持在计算图上（不 detach）
                act_var  = h.pop_var()
                act_mean_batch_tta.append(act_mean)
                act_var_batch_tta.append(act_var)
                # 清理该 hook 的缓冲，防止前后 batch 相互污染 / 占用内存
                h.clear()
                h.int_mean = []
                h.int_var = []

            # 卸载 hooks
            for hndl in hooks_list_tta:
                hndl.remove()

            # 计算 L1 loss（逐层、逐位置的 L1）
            loss_mean = torch.tensor(0., device=args.device)
            loss_var = torch.tensor(0., device=args.device)
            for i in range(n_chosen_layers):
                # clean_mean[i], clean_var[i] 在前面已 detach 成常量，所以只有 act_* 会反传梯度到模型
                loss_mean += l1_loss(act_mean_batch_tta[i].to(args.device), clean_mean[i].to(args.device))
                loss_var  += l1_loss(act_var_batch_tta[i].to(args.device),  clean_var[i].to(args.device))
            loss = (loss_mean + loss_var) * 0.5

            # 反向并更新全部参数
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()

    # 评估部分保持不变
    if dataset == 'plantdata':
        train_loader = create_plantdata(root=datastes_path, split='train', batch_size=batch_size, transforms=transforms_clean, proportion=0.7)
    else:
        train_loader, _ = create_celeba(root=datastes_path, split='train', batch_size=batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'], transforms=transforms_clean, max_samples=None)

    mymodel.eval()
    ood_results = evaluate_model(mymodel, te_loader, task_names, args.device)
    id_results = evaluate_model(mymodel, train_loader, task_names, args.device)

    result_line = "\t".join([
        f"{id_results[task]*100:.2f}\t{ood_results[task]*100:.2f}"
        for task in task_names
    ]) + "\n"
    f.write(result_line)
    f.flush()
    return

if __name__ == "__main__":

    transforms_clean = data_transforms(corruption=None) # 不添加噪声
    gaussian_corruption = data_transforms(corruption=gaussian_noise, severity=5) # 添加 Gaussian 噪声
    shot_corruption = data_transforms(corruption=shot_noise, severity=5) # 添加 Shot 噪声
    impulse_corruption = data_transforms(corruption=impulse_noise, severity=5) # 添加 Impulse 噪声
    defocus_corruption = data_transforms(corruption=defocus_blur, severity=5) # 添加 defocus_blur
    brightness_corruption = data_transforms(corruption=brightness, severity=5) # 添加 brightness
    contrast_corruption = data_transforms(corruption=contrast, severity=5) # 添加 contrast

    corruption_method = [gaussian_corruption, shot_corruption, impulse_corruption, defocus_corruption, brightness_corruption, contrast_corruption]
    corruption_names = ["Gaussian Noise", "Shot Noise", "Impulse Noise", "Defocus Blur", "Brightness", "Contrast"]

    plantdata_task_classes_name = ['CropType', 'Pathogen', 'Disease']
    celeba_task_classes_name = ['Attractive', 'Male', 'Smiling', 'Wearing Lipstick']

    # Output file's name
    filename = "results_ActMAD.txt"

    # task_names = plantdata_task_classes_name  # ['CropType', 'Pathogen', 'Disease']
    task_names = celeba_task_classes_name  # ['Attractive', 'Male', 'Smiling', 'Wearing Lipstick']

    with open('CelebA_results/'+filename, 'a', encoding='utf-8') as f:
        for transforms_corruption, corruption_name in zip(corruption_method, corruption_names):

            # Write separator marker
            f.write(f"\n=== TTA in {corruption_name} ===\n")

            # First line: task names + arrow
            header1 = "\t\t".join([f"{task} ↑" for task in task_names]) + "\n"

            # Second line: each task has ID and OOD columns
            header2 = "\t".join(["ID\tOOD"] * len(task_names)) + "\n"

            f.write(header1)
            f.write(header2)

            # PlantData
            # mymodel = MyModel(task_classes=plantdata_task_classes, device=device)
            # mymodel.load_state_dict(torch.load(filedir + 'resnet18_plantdata_7_3.pth'))
            # actmad_pipeline(mymodel, corruption_name, dataset='plantdata')

            # CelebA
            mymodel = MyModel(task_classes=celeba_task_classes, device=args.device)
            mymodel.load_state_dict(torch.load(filedir + 'resnet18_celeba.pth'))
            actmad_pipeline(mymodel, corruption_name, dataset='celeba')
