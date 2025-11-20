import os
import time
import json
import gc
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from data import create_kfold_splits, get_dataloader
from model import CNN

# 为了分析，我在代码中引入了wandb库可视化的分析
# 如果在执行时希望同样载入，请运行exper.py并登录自己的wandb账户，通过API key载入
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=True):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and device.type == 'cuda':
            # 使用新的 AMP 接口，避免 FutureWarning
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # AMP 下进行梯度裁剪需要先 unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples

def train_kfold(data_root='data/CUB_200_2011', epochs=15, batch_size=16, lr=1e-3, num_folds=10):
    """十折交叉验证，采用轻量设置以降低显存占用，并保存过程性结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 仅运行一次划分
    create_kfold_splits(data_root, n_splits=num_folds)

    # 回退到更稳定的预处理（之前实验更优）：固定尺寸 + 轻量翻转
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 运行目录（保存日志、模型与汇总）
    run_dir = os.path.join('runs', time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    # 记录全局配置，方便写报告
    dummy_model = CNN(num_classes=200)
    total_params = sum(p.numel() for p in dummy_model.parameters())
    config = {
        'data_root': data_root,
        'num_folds': num_folds,
        'epochs_per_fold': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'device': device.type,
        'input_resize': 160,
        'input_crop': 128,
        'augmentations': {
            'RandomHorizontalFlip': 0.5,
        },
        'weight_decay': 5e-4,
        'lr_scheduler': 'CosineAnnealingLR',
        'early_stopping_patience': 14,
        'model': {
            'type': 'LightDSCNN_4Conv',
            'channels': [24, 48, 96, 128],
            'depthwise_separable_from_block2': True,
            'dropout': 0.4,
            'adaptive_avg_pool': True,
            'total_params': int(total_params),
        },
        'loss': {
            'type': 'CrossEntropyLoss',
            'label_smoothing': 0.1,
        },
        'optimizer': 'AdamW',
        'weight_decay': 0.0005,
        'early_stopping_patience': 1,
        'amp': True,
        'lr_scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': epochs,
        },
    }
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    # 初始化 wandb 运行（如果可用）
    wandb_run = None
    if _WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=os.getenv('WANDB_PROJECT', 'cnn-experiments'),
                name=f"kfold-{num_folds}-{time.strftime('%Y%m%d_%H%M%S')}",
                config=config,
                reinit=True,
            )
        except Exception:
            wandb_run = None

    fold_val_accs = []

    for fold in range(num_folds):
        print(f"\n===== Fold {fold+1}/{num_folds} =====")

        train_loader, val_loader = get_dataloader(
            data_root=data_root,
            fold_idx=fold,
            transform=transform,
            batch_size=batch_size,
            num_workers=0,  # Windows 下多进程加载可能占用更多内存
            pin_memory=(device.type == 'cuda'),
        )

        # 模型、损失函数、优化器
        model = CNN(num_classes=200).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        patience, bad_epochs = 14, 0  # 提高早停容忍度，尽量跑满 15 个 epoch 观察曲线
        fold_metrics = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=True)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            # 记录当前学习率并步进调度器
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # 记录每个 epoch 的指标
            fold_metrics.append({
                'fold': fold,
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'lr': float(current_lr),
            })
            if wandb_run is not None:
                epoch_global = fold * epochs + (epoch + 1)
                wandb.log({
                    'fold': fold,
                    'epoch': epoch + 1,
                    'train/loss': float(train_loss),
                    'train/acc': float(train_acc),
                    'val/loss': float(val_loss),
                    'val/acc': float(val_acc),
                    'lr': float(current_lr),
                }, step=epoch_global)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                bad_epochs = 0
                # 保存当前折的最佳模型
                fold_dir = os.path.join(run_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)
                torch.save({
                    'state_dict': model.state_dict(),
                    'best_val_acc': float(best_val_acc),
                }, os.path.join(fold_dir, 'best_model.pth'))
            else:
                bad_epochs += 1
                if bad_epochs > patience:
                    print("Early stopping triggered.")
                    break

        # 保存当前折的训练曲线
        fold_dir = os.path.join(run_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        pd.DataFrame(fold_metrics).to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)

        fold_val_accs.append(best_val_acc)

        # 释放显存/内存，防止累积导致溢出
        del model, optimizer, criterion, scaler, scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print("\n===== Cross-Validation Results =====")
    for i, acc in enumerate(fold_val_accs):
        print(f"Fold {i+1}: best val acc = {acc:.4f}")
    mean_acc = mean(fold_val_accs)
    print(f"Mean Acc: {mean_acc:.4f}")

    # 保存汇总结果
    summary = {
        'fold_best_val_accs': [float(a) for a in fold_val_accs],
        'mean_best_val_acc': float(mean_acc),
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if wandb_run is not None:
        wandb_run.summary['fold_best_val_accs'] = [float(a) for a in fold_val_accs]
        wandb_run.summary['mean_best_val_acc'] = float(mean_acc)
        wandb_run.finish()


if __name__ == "__main__":
    # 默认使用更保守的配置以避免 3090 laptop 显卡过载
    train_kfold()