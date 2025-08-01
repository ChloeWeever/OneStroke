import matplotlib.pyplot as plt
import numpy as np
from evaluate import Evaluator
from pathlib import Path


def plot_accuracy(model_path):
    evaluator = Evaluator()
    accuracies = []
    char_ids = range(40)  # 字符编号 0-39

    for char_id in char_ids:
        target_path = Path(f"../data/output_img/{char_id}/0")
        result_path = Path(f"../data/output_img/{char_id}/0/0.jpg")

        try:
            accuracy = evaluator.main(target_path, model_path, result_path)
            accuracies.append(accuracy['total_accuracy'])
            print(f"Character {char_id} accuracy: {accuracy['total_accuracy']:.2f}%")
        except Exception as e:
            print(f"Error processing character {char_id}: {str(e)}")
            accuracies.append(0)  # 出错时记为0

    # 计算平均准确率
    mean_accuracy = np.mean(accuracies)

    # 创建图表（纯英文）
    plt.figure(figsize=(12, 6))
    plt.plot(char_ids, accuracies,
             marker='o', linestyle='-', color='b',
             label=f'Single-character accuracy (Avg: {mean_accuracy:.2f}%)')

    # 平均线
    plt.axhline(y=mean_accuracy, color='r', linestyle='--',
                linewidth=2, label='Average accuracy')

    plt.text(40.5, mean_accuracy, f'{mean_accuracy:.2f}%',
             color='r', va='center', ha='left', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 图表标题和标签（英文）
    plt.title(f"UNet Model Prediction Accuracy", fontsize=14)
    plt.xlabel("Character ID (0-39)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # 保存图表
    output_path = f"{Path(model_path).stem}_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    model_path = Path("../models/model_5coder/unet_model.pth")
    plot_accuracy(model_path)