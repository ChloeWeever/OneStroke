from typing import Optional, Callable, Union

import matplotlib.pyplot as plt
import numpy as np

from evaluate_.evaluate import Evaluator
from pathlib import Path


def plot_accuracy(weight_path: Optional[Union[str, Path, object]] = None, predict_fn: Optional[Callable[[Union[str, Path], float], np.ndarray]] = None):
    evaluator = Evaluator()
    accuracies = []
    style_accuracies = []
    char_ids = range(40)  # 字符编号 0-39

    for char_id in char_ids:
        for i in (18, 19):
            ture_mask_path = Path(f"data/output_img/{char_id}/{i}")
            result_path = Path(f"data/output_img/{char_id}/{i}/0.jpg")

            try:
                accuracy = evaluator.main(ture_mask_path, weight_path, result_path, threshold=0.5, predict_fn=predict_fn)
                accuracies.append(accuracy['total_accuracy'])
                print(f"Character {char_id} accuracy: {accuracy['total_accuracy']:.2f}%")
            except Exception as e:
                print(f"Error processing character {char_id}: {str(e)}")
                accuracies.append(0)  # 出错时记为0

        mean_accuracy = np.mean(accuracies)
        style_accuracies.append(mean_accuracy)
        accuracies.clear()
    # 计算平均准确率
    mean_style_accuracy = np.mean(style_accuracies)

    random_accuracies = []
    for char_id in (40, 41, 42):
        for i in range(18):
            ture_mask_path = Path(f"data/output_img/{char_id}/{i}")
            result_path = Path(f"data/output_img/{char_id}/{i}/0.jpg")

            try:
                accuracy = evaluator.main(ture_mask_path, model_path, result_path, threshold=0.5)
                accuracies.append(accuracy['total_accuracy'])
                print(f"Character {char_id} accuracy: {accuracy['total_accuracy']:.2f}%")
            except Exception as e:
                print(f"Error processing character {char_id}: {str(e)}")
                accuracies.append(0)  # 出错时记为0

        mean_accuracy = np.mean(accuracies)
        random_accuracies.append(mean_accuracy)
        accuracies.clear()
        # 计算平均准确率
    mean_random_accuracy = np.mean(style_accuracies)

    return {
        "mean_style_accuracy": mean_style_accuracy,
        "mean_random_accuracy": mean_random_accuracy
    }

