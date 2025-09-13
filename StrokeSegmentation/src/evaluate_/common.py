import torch.nn.functional as F
import torch
from core.config import settings
from predictor.transunet_predictor import transUnetPredictor
from predictor.unet_predictor import UNetPredictor
from predictor.fcn_predictor import FCNPredictor
from predictor.deeplab_predictor import deeplabv3Predictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    white_pixels = np.all(img_array >= 240, axis=-1)
    mask_o = np.where(white_pixels, 0, 1).astype(np.uint8)
    mask_o = np.expand_dims(mask_o, axis=-1)
    mask_o = np.repeat(mask_o, 6, axis=-1)
    # print("mask_o shape: "+str(mask_o.shape))
    if settings.MODEL == "unet":
        # 使用UNet模型进行预测（原始代码）
        # print("\n使用UNet模型进行预测:")
        unet_predictor = UNetPredictor(settings.PREDICT_MODEL, settings.DEVICE)
        unet_result = unet_predictor.predict(img_path)
        unet_result = np.logical_and(unet_result, mask_o)
        return unet_result
    elif settings.MODEL == "fcn":
        # 使用FCN模型进行预测
        # print("\n使用FCN模型进行预测:")
        try:
            fcn_predictor = FCNPredictor(settings.PREDICT_MODEL, settings.DEVICE)
            fcn_result = fcn_predictor.predict(img_path)
            # print(f"FCN预测结果形状: {fcn_result.shape}")
            fcn_result = np.logical_and(fcn_result, mask_o)
            return fcn_result
        except Exception as e:
            print(f"FCN模型预测过程中出现错误: {e}")
            print("请确保FCN模型文件 'models/fcn_model_new.pth' 存在")

    elif settings.MODEL == "transunet":
        # 使用TransUNet模型进行预测
        # print("\n使用TransUNet模型进行预测:")
        try:
            transunet_predictor = transUnetPredictor(
                settings.PREDICT_MODEL, settings.DEVICE
            )
            transunet_result = transunet_predictor.predict(img_path)
            # print(f"TransUNet预测结果形状: {transunet_result.shape}")
            transunet_result = np.logical_and(transunet_result, mask_o)
            return transunet_result

        except Exception as e:
            print(f"TransUNet模型预测过程中出现错误: {e}")
            print("请确保TransUNet模型文件存在")

    elif settings.MODEL == "deeplabv3":
        # 使用DeepLabV3+模型进行预测
        # print("\n使用DeepLabV3+模型进行预测:")
        try:
            deeplab_predictor = deeplabv3Predictor(
                settings.PREDICT_MODEL, settings.DEVICE
            )
            deeplab_result = deeplab_predictor.predict(img_path)
            # print(f"DeepLabV3+预测结果形状: {deeplab_result.shape}")
            deeplab_result = np.logical_and(deeplab_result, mask_o)
            return deeplab_result

        except Exception as e:
            print(f"DeepLabV3+模型预测过程中出现错误: {e}")
            print("请确保DeepLabV3+模型文件 'models/deeplab_model_new.pth' 存在")
    print(f"预测过程中出现错误: {e}")
    return None


def BCE_Dice():
    # 用于存储所有图片的bce和dice值
    all_bce_values = []
    all_dice_values = []
    image_indices = []
    
    for i in range(0, 40):
        print(f"正在处理第{i}个汉字")
        for k in [19, 20]:
            try:
                predict_result = predict(f"data/output_img/{i}/{k}/0.jpg")
                predict_result = torch.from_numpy(predict_result).permute(2, 0, 1).float()
                mask_path = f"data/output_img/{i}/{k}/0.npy"
                mask = np.load(mask_path)
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()
                bce = get_bce(predict_result, mask)
                dice = get_dice(predict_result, mask)
                
                # 存储结果
                all_bce_values.append(bce.item())  # 使用.item()获取标量值
                all_dice_values.append(dice.item())
                image_indices.append(f"{i},{k-19}")
                
                # print(f"bce: {bce}, dice: {dice}")
            except Exception as e:
                print(f"处理图片{i}_{k}时出错: {e}")
    
    # 计算平均值
    avg_bce = np.mean(all_bce_values) if all_bce_values else 0
    avg_dice = np.mean(all_dice_values) if all_dice_values else 0
    
    print(f"\n平均值: BCE = {avg_bce:.4f}, Dice = {avg_dice:.4f}")
    
    # 绘制折线图 - 合并到同一张图中
    plt.figure(figsize=(20, 3))
    
    # 创建两个y轴，因为BCE和Dice的范围可能不同
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 在第一个y轴上绘制BCE
    bce_line, = ax1.plot(image_indices, all_bce_values, 'o-', color='blue', label=f'BCE Loss (Avg: {avg_bce:.4f})')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('BCE Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.tick_params(axis='x', labelsize=5)
    
    # 在第二个y轴上绘制Dice
    dice_line, = ax2.plot(image_indices, all_dice_values, 's-', color='green', label=f'Dice Coefficient (Avg: {avg_dice:.4f})')
    ax2.set_ylabel('Dice Value', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax1.tick_params(axis='x', labelsize=5)
    
    # 添加水平线表示平均值
    ax1.axhline(y=avg_bce, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=avg_dice, color='green', linestyle='--', alpha=0.5)
    
    # 设置标题和网格
    plt.title('BCE Loss and Dice Coefficient Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 合并图例
    lines = [bce_line, dice_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='best')

    plt.tight_layout()
    
    # 保存和显示图像
    plt.savefig('bce_dice_combined_plot.png')  # 保存为新的文件名
    plt.show()

def get_bce(predictions, targets):
    bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction="mean")
    return bce


def get_dice(predictions, targets):
    # 将logits转换为概率
    preds = torch.sigmoid(predictions)

    # 计算每个类别的Dice系数
    smooth = 1e-6  # 防止除零错误
    dice_scores = []

    # 遍历每个类别
    for i in range(predictions.size(0)):
        # 计算交集
        intersection = (preds[i] * targets[i]).sum()

        # 计算并集（用于Dice系数计算）
        union = preds[i].sum() + targets[i].sum()

        # 计算Dice系数
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    # 返回所有类别的平均Dice系数
    return torch.mean(torch.tensor(dice_scores))
