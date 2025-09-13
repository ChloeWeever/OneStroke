import torch
from torchvision import transforms
from model.other.transUNet.transUnet_model import TransUNet
from PIL import Image

class transUnetPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # 根据需要可以选择TransUNet, TransUNetSmall或TransUNetLarge
        self.model = TransUNet(n_channels=3, n_classes=6, img_size=500).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"加载TransUNet模型时出错: {e}")
            print("请确保模型文件存在且格式正确")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, threshold=0.5):
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.sigmoid(output)
            binary_output = (probabilities > threshold).float()

        result = binary_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return result