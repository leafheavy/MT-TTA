import torch
import torch.optim as optim

from model import MyModel
from utils import *
from create_data import create_celeba, data_transforms
from corruption_method import *
from GradientConsensus import gradient_consensus
from Fishers import *

# import adaptation algorithms
from M_EATA import EATA_MultiTask
from M_SAR import SAR_MultiTask
from CoCo import CoCo

if __name__ == "__main__":

    args = parse_args()
    set_seed(args.random_seed)

    transforms_clean = data_transforms(corruption=None) # 不添加噪声
    gaussian_corruption = data_transforms(corruption=gaussian_noise, severity=5) # 添加 Gaussian 噪声
    shot_corruption = data_transforms(corruption=shot_noise, severity=5) # 添加 Shot 噪声
    impulse_corruption = data_transforms(corruption=impulse_noise, severity=5) # 添加 Impulse 噪声
    defocus_corruption = data_transforms(corruption=defocus_blur, severity=5) # 添加 defocus_blur
    brightness_corruption = data_transforms(corruption=brightness, severity=5) # 添加 brightness
    contrast_corruption = data_transforms(corruption=contrast, severity=5) # 添加 contrast

    corruption_method = [gaussian_corruption, shot_corruption, impulse_corruption, defocus_corruption, brightness_corruption, contrast_corruption]
    corruption_names = ["Gaussian Noise", "Shot Noise", "Impulse Noise", "Defocus Blur", "Brightness", "Contrast"] 

    train_loader, task_names = create_celeba(root=datastes_path, split='train', batch_size=args.batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'], transforms=transforms_clean, max_samples=None)
    filename = "CoCo_results.txt" # CoCo

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

            test_loader, _ = create_celeba(root=datastes_path, split='test', batch_size=args.batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'], transforms=transforms_corruption, max_samples=None)

            # Reload pre-trained model during each corruption
            mymodel = MyModel(task_classes=celeba_task_classes, device=args.device)
            mymodel.load_state_dict(torch.load(filedir + 'resnet18_celeba.pth'))

            mymodel = configure_model(mymodel)
            print(f'Number of task heads: {len(mymodel.task_heads)}')

            tta_params = [p for p in mymodel.parameters() if p.requires_grad]
            tta_optimizer = optim.SGD(tta_params, lr=args.lr_tta, momentum=0.9)

            fisher_loader, _ = create_celeba(root=datastes_path, split='train', batch_size=args.batch_size, selected_attrs=['Attractive', 'Male', 'Smiling', 'Wearing_Lipstick'], transforms=transforms_clean, max_samples=10000) # 用于计算 Fisher 矩阵的数据加载器 (源域)
            fishers_dict = compute_fisher_matrix(mymodel, fisher_loader)

            # Initial TTA setting
            coco_model = CoCo(
                model=mymodel,
                optimizer=tta_optimizer,
                steps=1,
                episodic=False,                    # Offline adaptation
                tta_params=tta_params,
                grad_adjust_fn=gradient_consensus, # enable Gradient Consus (GC)
                require_fishers=True,              # enable Plasticity Constraints (PC)
                fishers=fishers_dict,
                fisher_alpha=args.fisher_alpha,
                specific_task=None
            )

            for tta_step in tqdm(range(args.tta_steps), desc=f"TTA in {corruption_name}"):
                # forward and adapt
                coco_model.steps = 1
                for iter_, (inputs, _) in enumerate(test_loader, start=1):
                    
                    inputs = inputs.to(torch.float32).to(args.device)

                    outputs = coco_model(inputs)
            
                # Evaluating
                coco_model.steps = 0
                ood_results, avg_task_losses, total_loss = evaluate_model(coco_model, test_loader, task_names, loss_func=None, require_loss=True, device=args.device)

                print(f"Task-wise Losses: {avg_task_losses}\tTotal Loss: {total_loss}")

                id_results = evaluate_model(coco_model, train_loader, task_names, loss_func=None, require_loss=False, device=args.device)
                
                # F1_{ID} and F1_{OOD}
                result_line = "\t".join([
                    f"{id_results[task]*100:.2f}\t{ood_results[task]*100:.2f}"
                    for task in task_names
                ]) + "\n"
                f.write(result_line)
                # f.write(f"\n=== TTA in {tta_step+1}-th steps ===\n")
                f.flush()
            f.write("\n")