"""
train_srn.py - SRN训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np
from .srn_model import SiameseRegressionNetwork, SRNLoss
from code.utils.dataset_srn import CalligraphyDataset
from code.utils.global_config import Config
from evaluation.metrics import calculate_mae, calculate_pcc

def train_srn(config_path='configs/training_config.yaml'):
    """
    训练SRN模型
    
    Args:
        config_path: 配置文件路径
        
    Raises:
        FileNotFoundError: 配置文件不存在或数据目录不存在
        ValueError: 配置参数无效
        RuntimeError: 训练准备或执行过程失败
        
    Returns:
        Tuple[str, float]: 保存的最佳模型路径和对应的验证损失
    """
    def _validate_config(config):
        """验证配置参数的完整性"""
        required_params = [
            'srn.backbone', 'srn.feature_dims', 'srn.dropout_rate',
            'srn.learning_rate', 'srn.batch_size', 'srn.epochs',
            'srn.scheduler.step_size', 'srn.scheduler.gamma',
            'data.data_dir'
        ]
        missing_params = []
        for param in required_params:
            if not config.has(param):
                missing_params.append(param)
        if missing_params:
            raise ValueError(f"缺少必要的配置参数: {', '.join(missing_params)}")
            
    def _setup_datasets(config):
        """初始化训练和验证数据集"""
        data_dir = config.get('data.data_dir')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        train_dataset = CalligraphyDataset(
            data_dir=data_dir,
            split='train',
            transform=None
        )
        
        val_dataset = CalligraphyDataset(
            data_dir=data_dir,
            split='val',
            transform=None
        )
        
        return train_dataset, val_dataset
    
    try:
        # 1. 验证配置文件
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 2. 加载和验证配置
        config = Config(config_path)
        _validate_config(config)
        
        # 3. 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 4. 创建模型
        model = SiameseRegressionNetwork(
            backbone_name=config.get('srn.backbone'),
            pretrained=config.get('srn.pretrained', True),
            feature_dims=config.get('srn.feature_dims'),
            dropout_rate=config.get('srn.dropout_rate')
        ).to(device)
        
        # 5. 初始化训练组件
        criterion = SRNLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('srn.learning_rate'),
            weight_decay=config.get('srn.weight_decay', 0.0001)
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('srn.scheduler.step_size'),
            gamma=config.get('srn.scheduler.gamma')
        )
        
        # 6. 准备数据集
        train_dataset, val_dataset = _setup_datasets(config)
        
    except FileNotFoundError as e:
        print(f"文件路径错误: {str(e)}")
        raise
    except ValueError as e:
        print(f"配置参数错误: {str(e)}")
        raise
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        raise RuntimeError(f"训练准备阶段失败: {str(e)}")
    
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('srn.batch_size'),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('srn.batch_size'),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True if torch.cuda.is_available() else False
        )
    except Exception as e:
        raise RuntimeError(f"数据加载器创建失败: {str(e)}")
    
    # 训练循环
    best_val_loss = float('inf')
    num_epochs = config.get('srn.epochs')
    patience = config.get('srn.early_stopping.patience', 10)
    min_delta = config.get('srn.early_stopping.min_delta', 1e-4)
    no_improve_count = 0
    
    # 创建保存目录
    model_save_dir = os.path.join(config.get('model_save_dir', 'models'), 'srn')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 记录训练日志
    log_file = os.path.join(model_save_dir, 'training.log')
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_mae,train_pcc,val_loss,val_mae,val_pcc\n")
    
    # 规范化函数
    def normalize_images(images):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        return images.float() / 255.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_pcc = 0.0
        
        train_progress = tqdm(train_loader, desc='Training')
        for batch_idx, (templates, copies, labels) in enumerate(train_progress):
            try:
                # 数据预处理
                templates = normalize_images(templates).to(device)
                copies = normalize_images(copies).to(device)
                labels = labels.to(device, dtype=torch.float32)
                
                # 清零梯度
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零方式
                
                # 前向传播
                template_outputs, copy_outputs = model(templates, copies)
                
                # 计算损失
                loss = criterion(template_outputs, copy_outputs, labels)
                
                if not torch.isfinite(loss):
                    raise ValueError("损失值无效(NaN或Inf)")
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                # 计算指标
                with torch.no_grad():
                    predictions = 10 - torch.norm(template_outputs - copy_outputs, dim=1)
                    mae = calculate_mae(predictions.cpu().numpy(), labels.cpu().numpy())
                    pcc, _ = calculate_pcc(predictions.cpu().numpy(), labels.cpu().numpy())
                
                # 累积指标
                batch_size = templates.size(0)
                train_loss += loss.item() * batch_size
                train_mae += mae * batch_size
                train_pcc += pcc * batch_size
                
                # 更新进度条
                train_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{mae:.4f}',
                    'PCC': f'{pcc:.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"\nWARNING: GPU内存不足，跳过当前batch")
                    continue
                raise
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        total_val_samples = 0
        
        val_progress = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for batch_idx, (templates, copies, labels) in enumerate(val_progress):
                try:
                    # 数据预处理
                    templates = normalize_images(templates).to(device)
                    copies = normalize_images(copies).to(device)
                    labels = labels.to(device, dtype=torch.float32)
                    
                    # 前向传播
                    template_outputs, copy_outputs = model(templates, copies)
                    
                    # 计算损失
                    loss = criterion(template_outputs, copy_outputs, labels)
                    
                    if not torch.isfinite(loss):
                        print(f"\nWARNING: 验证损失无效，跳过当前batch")
                        continue
                    
                    # 计算预测值
                    predictions = 10 - torch.norm(template_outputs - copy_outputs, dim=1)
                    
                    # 累积损失和收集预测结果
                    batch_size = templates.size(0)
                    val_loss += loss.item() * batch_size
                    total_val_samples += batch_size
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
                    # 更新进度条
                    val_progress.set_postfix({'Loss': f'{loss.item():.4f}'})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"\nWARNING: GPU内存不足，跳过当前batch")
                        continue
                    raise
        
        # 计算平均指标
        train_samples = len(train_loader.dataset)
        train_loss /= train_samples
        train_mae /= train_samples
        train_pcc /= train_samples
        
        val_loss /= total_val_samples
        val_predictions = np.array(val_predictions)
        val_labels = np.array(val_labels)
        val_mae = calculate_mae(val_predictions, val_labels)
        val_pcc, _ = calculate_pcc(val_predictions, val_labels)
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练日志
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_mae:.6f},{train_pcc:.6f},"
                   f"{val_loss:.6f},{val_mae:.6f},{val_pcc:.6f}\n")
        
        # 打印结果
        print(f"训练损失: {train_loss:.4f}, MAE: {train_mae:.4f}, PCC: {train_pcc:.4f}")
        print(f"验证损失: {val_loss:.4f}, MAE: {val_mae:.4f}, PCC: {val_pcc:.4f}")
        
        # 早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_pcc': val_pcc,
                'config': config.to_dict()
            }, best_model_path)
            print(f"最佳模型已保存，验证损失: {best_val_loss:.4f}")
        else:
            no_improve_count += 1
            
        # 定期保存检查点
        save_interval = config.get('save_interval', 10)
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"检查点已保存: checkpoint_epoch_{epoch+1}.pth")
            
        # 早停判断
        if no_improve_count >= patience:
            print(f"\n验证损失在{patience}个epoch内未改善，停止训练")
            break
            
        # 更新学习率
        scheduler.step()
    
    print(f"\nSRN训练完成")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存在: {os.path.join(model_save_dir, 'best_model.pth')}")
    
    return os.path.join(model_save_dir, 'best_model.pth'), best_val_loss

'''
# 用于测试的示例代码
if __name__ == "__main__":
    try:
        # 注意：需要先创建配置文件和数据集
        model_path, best_loss = train_srn('configs/training_config.yaml')
        print(f"训练完成！最佳模型已保存到: {model_path}")
        print(f"最佳验证损失: {best_loss:.4f}")
    except Exception as e:
        print(f"训练失败: {str(e)}")
'''