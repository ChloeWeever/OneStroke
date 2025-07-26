import re
import matplotlib.pyplot as plt
from typing import List, Tuple


def parse_losses(log_text: str) -> Tuple[List[float], List[float]]:
    """Parse train and validation losses from the log text"""
    train_losses = []
    val_losses = []

    # Regular expression to match loss lines
    pattern = r"Train Loss: ([\d.]+)\nVal Loss: ([\d.]+)"

    matches = re.findall(pattern, log_text)

    for train_loss, val_loss in matches:
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

    return train_losses, val_losses


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """Plot training and validation loss curves"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Find and mark the best validation loss
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    plt.scatter(best_epoch, best_val_loss, s=100, c='green', marker='o', label=f'Best Val Loss: {best_val_loss:.6f}')

    plt.grid(True)
    #plt.show()
    # 保存图片
    plt.savefig('loss_plot.png')


if __name__ == "__main__":
    # Example usage with your log (you can replace this with reading from a file)
    log_text = """
D:\Anaconda\python.exe F:\OneStroke\StrokeSegmentation\src\main.py
path: F:\OneStroke\StrokeSegmentation\src
D:\Anaconda\Lib\site-packages\torch\functional.py:554: UserWarning:

torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\TensorShape.cpp:4316.)

Epoch 0/49
----------
Train Loss: 0.00832367
Val Loss: 0.00730606
Epoch 1/49
----------
Train Loss: 0.00698244
Val Loss: 0.00598599
Epoch 2/49
----------
Train Loss: 0.00635670
Val Loss: 0.00614112
Epoch 3/49
----------
Train Loss: 0.00595444
Val Loss: 0.00538433
Epoch 4/49
----------
Train Loss: 0.00549144
Val Loss: 0.00501068
Epoch 5/49
----------
Train Loss: 0.00517037
Val Loss: 0.00474956
Epoch 6/49
----------
Train Loss: 0.00470561
Val Loss: 0.00418443
Epoch 7/49
----------
Train Loss: 0.00441495
Val Loss: 0.00351185
Epoch 8/49
----------
Train Loss: 0.00421800
Val Loss: 0.00420056
Epoch 9/49
----------
Train Loss: 0.00379997
Val Loss: 0.00363251
Model saved at epoch 10 as ../models/checkpoint/unet_model_epoch_10.pth
Epoch 10/49
----------
Train Loss: 0.00363124
Val Loss: 0.00288207
Epoch 11/49
----------
Train Loss: 0.00333644
Val Loss: 0.00319263
Epoch 12/49
----------
Train Loss: 0.00326299
Val Loss: 0.00312984
Epoch 13/49
----------
Train Loss: 0.00310795
Val Loss: 0.00256456
Epoch 14/49
----------
Train Loss: 0.00293718
Val Loss: 0.00253833
Epoch 15/49
----------
Train Loss: 0.00276010
Val Loss: 0.00298998
Epoch 16/49
----------
Train Loss: 0.00255798
Val Loss: 0.00247621
Epoch 17/49
----------
Train Loss: 0.00237180
Val Loss: 0.00212043
Epoch 18/49
----------
Train Loss: 0.00248808
Val Loss: 0.00207004
Epoch 19/49
----------
Train Loss: 0.00217117
Val Loss: 0.00207171
Model saved at epoch 20 as ../models/checkpoint/unet_model_epoch_20.pth
Epoch 20/49
----------
Train Loss: 0.00215072
Val Loss: 0.00215039
Epoch 21/49
----------
Train Loss: 0.00205009
Val Loss: 0.00181658
Epoch 22/49
----------
Train Loss: 0.00185780
Val Loss: 0.00212393
Epoch 23/49
----------
Train Loss: 0.00191483
Val Loss: 0.00216008
Epoch 24/49
----------
Train Loss: 0.00179727
Val Loss: 0.00166344
Epoch 25/49
----------
Train Loss: 0.00176661
Val Loss: 0.00275198
Epoch 26/49
----------
Train Loss: 0.00172765
Val Loss: 0.00176813
Epoch 27/49
----------
Train Loss: 0.00167875
Val Loss: 0.00212793
Epoch 28/49
----------
Train Loss: 0.00142659
Val Loss: 0.00136280
Epoch 29/49
----------
Train Loss: 0.00128133
Val Loss: 0.00134589
Model saved at epoch 30 as ../models/checkpoint/unet_model_epoch_30.pth
Epoch 30/49
----------
Train Loss: 0.00125632
Val Loss: 0.00141811
Epoch 31/49
----------
Train Loss: 0.00125702
Val Loss: 0.00137019
Epoch 32/49
----------
Train Loss: 0.00126814
Val Loss: 0.00135752
Epoch 33/49
----------
Train Loss: 0.00123927
Val Loss: 0.00135885
Epoch 34/49
----------
Train Loss: 0.00122754
Val Loss: 0.00135067
Epoch 35/49
----------
Train Loss: 0.00121707
Val Loss: 0.00136235
Epoch 36/49
----------
Train Loss: 0.00119658
Val Loss: 0.00136414
Epoch 37/49
----------
Train Loss: 0.00123598
Val Loss: 0.00135314
Epoch 38/49
----------
Train Loss: 0.00119828
Val Loss: 0.00136818
Epoch 39/49
----------
Train Loss: 0.00120234
Val Loss: 0.00136567
Model saved at epoch 40 as ../models/checkpoint/unet_model_epoch_40.pth
Epoch 40/49
----------
Train Loss: 0.00119771
Val Loss: 0.00135437
Epoch 41/49
----------
Train Loss: 0.00121579
Val Loss: 0.00136878
Epoch 42/49
----------
Train Loss: 0.00122817
Val Loss: 0.00138416
Epoch 43/49
----------
Train Loss: 0.00120339
Val Loss: 0.00135259
Epoch 44/49
----------
Train Loss: 0.00119638
Val Loss: 0.00139620
Epoch 45/49
----------
Train Loss: 0.00122595
Val Loss: 0.00134812
Epoch 46/49
----------
Train Loss: 0.00119951
Val Loss: 0.00134853
Epoch 47/49
----------
Train Loss: 0.00119229
Val Loss: 0.00135078
Epoch 48/49
----------
Train Loss: 0.00123157
Val Loss: 0.00134694
Epoch 49/49
----------
Train Loss: 0.00120191
Val Loss: 0.00136635
Model saved at epoch 50 as ../models/checkpoint/unet_model_epoch_50.pth
Training complete in 105m 11s
Best val Loss: 0.00134589
Model saved as 'unet_model.pth'

Process finished with exit code 0

    """

    train_losses, val_losses = parse_losses(log_text)
    plot_losses(train_losses, val_losses)