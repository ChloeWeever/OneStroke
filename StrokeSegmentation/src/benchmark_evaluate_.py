import os
from evaluate_.benchmark_evaluate import * 
from predictor.unet_predictor import UNetPredictor
from predictor.deeplab_predictor import deeplabv3Predictor
from predictor.transunet_predictor import transUnetPredictor
from predictor.fcn_predictor import FCNPredictor
from core.config import settings

def predict(img_path,threshold):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    white_pixels = np.all(img_array >= 240, axis=-1)
    mask_o = np.where(white_pixels, 0, 1).astype(np.uint8)
    mask_o = np.expand_dims(mask_o, axis=-1)
    mask_o = np.repeat(mask_o, 6, axis=-1)
    model = UNetPredictor(model_path)
    result = model.predict(img_path, threshold=threshold)
    result = np.logical_and(result, mask_o)
    return result

if __name__ == "__main__":
    print(f"=================================================================================================")
    print(f"Working directory: {os.getcwd()}")
    print(settings)
    print(f"=================================================================================================")
    model_path = settings.EVALUATE_MODEL
    model = UNetPredictor(model_path)
    result = plot_accuracy(model_path, model.predict)
    print(result)