"""
main.py - 主程序入口
"""

import numpy as np
import cv2
import torch
import os
from code.preprocessing.image_preprocessing import preprocess_image
from code.preprocessing.image_registration import register_images
from code.features import extract_features_for_srafe
from code.srn.srn_model import SiameseRegressionNetwork
from code.fusion.feature_fusion import fuse_features
from code.fusion.regression import AestheticRegressor

def evaluate_calligraphy_aesthetic(template_path: str, 
                                  copy_path: str,
                                  srn_model_path: str = None,
                                  regressor_model_path: str = None) -> float:
    """
    评估书法美学质量
    
    Args:
        template_path: 模板图像路径
        copy_path: 摹本图像路径
        srn_model_path: SRN模型路径
        regressor_model_path: 回归器模型路径
        
    Returns:
        score: 美学评分 (0-10)
    """
    try:
        # 1. 加载图像
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"模板文件不存在: {template_path}")
        if not os.path.exists(copy_path):
            raise FileNotFoundError(f"摹本文件不存在: {copy_path}")
            
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        copy = cv2.imread(copy_path, cv2.IMREAD_GRAYSCALE)
        
        if template is None:
            raise ValueError(f"模板图像读取失败: {template_path}")
        if copy is None:
            raise ValueError(f"摹本图像读取失败: {copy_path}")
        
        # 2. 图像预处理
        template_processed = preprocess_image(template)
        copy_processed = preprocess_image(copy)
        
        # 3. 图像配准
        copy_registered, template_registered = register_images(copy_processed, template_processed)
        
        # 4. 提取手工特征
        features = extract_features_for_srafe(template_registered, copy_registered)
        
        # 5. 提取深度特征（如果提供了SRN模型）
        if srn_model_path and os.path.exists(srn_model_path):
            try:
                # 加载SRN模型
                srn_model = SiameseRegressionNetwork()
                srn_model.load_state_dict(torch.load(srn_model_path, map_location=torch.device('cpu')))
                srn_model.eval()
                
                # 转换图像为张量
                template_tensor = torch.from_numpy(template_registered).float().unsqueeze(0).unsqueeze(0) / 255.0
                copy_tensor = torch.from_numpy(copy_registered).float().unsqueeze(0).unsqueeze(0) / 255.0
                
                # 提取深度特征
                with torch.no_grad():
                    template_features, copy_features = srn_model(template_tensor, copy_tensor)
                    features['deep_features_template'] = template_features.squeeze().numpy()
                    features['deep_features_copy'] = copy_features.squeeze().numpy()
            except Exception as e:
                print(f"Warning: Failed to extract deep features: {e}")
        
        # 6. 特征融合
        fused_features = fuse_features(features)
        
        # 7. 回归预测
        if regressor_model_path and os.path.exists(regressor_model_path):
            try:
                regressor = AestheticRegressor(regressor_model_path)
                score = regressor.predict(fused_features)
            except Exception as e:
                print(f"Warning: Failed to use regressor model: {e}")
                # 如果回归器失败，使用简单的特征加权
                score = np.clip(np.mean(fused_features) * 10, 0, 10)
        else:
            # 如果没有回归器模型，使用简单的特征加权
            score = np.clip(np.mean(fused_features) * 10, 0, 10)
        
        return float(score)
        
    except Exception as e:
        print(f"Error in evaluate_calligraphy_aesthetic: {e}")
        # 返回默认评分
        return 5.0

def batch_evaluate(input_file: str, 
                  output_file: str,
                  srn_model_path: str = None,
                  regressor_model_path: str = None):
    """
    批量评估书法美学质量
    
    Args:
        input_file: 输入文件路径 (CSV格式，包含template_path和copy_path列)
        output_file: 输出文件路径
        srn_model_path: SRN模型路径
        regressor_model_path: 回归器模型路径
    """
    import pandas as pd
    
    # 加载输入数据
    df = pd.read_csv(input_file)
    
    # 检查必需的列
    required_columns = ['template_path', 'copy_path']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"输入文件缺少必需的列: {', '.join(missing_columns)}")
    
    # 添加预测结果列
    df['predicted_score'] = 0.0
    
    print(f"Starting batch evaluation of {len(df)} samples...")
    
    # 逐个预测
    for idx, row in df.iterrows():
        template_path = row['template_path']
        copy_path = row['copy_path']
        
        try:
            # 预测美学评分
            score = evaluate_calligraphy_aesthetic(
                template_path=template_path,
                copy_path=copy_path,
                srn_model_path=srn_model_path,
                regressor_model_path=regressor_model_path
            )
            
            df.loc[idx, 'predicted_score'] = score
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")
                
        except Exception as e:
            print(f"Error evaluating sample {copy_path}: {e}")
            df.loc[idx, 'predicted_score'] = 5.0  # 默认评分
    
    # 保存预测结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Batch evaluation completed. Results saved to: {output_file}")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SRAFE书法美学评估系统')
    parser.add_argument('--template', type=str, help='模板图像路径')
    parser.add_argument('--copy', type=str, help='摹本图像路径')
    parser.add_argument('--input-file', type=str, help='输入文件路径 (CSV格式)')
    parser.add_argument('--output-file', type=str, default='results/predictions.csv',
                       help='输出文件路径')
    parser.add_argument('--srn-model', type=str, default='models/srn_resnet101.pth',
                       help='SRN模型路径')
    parser.add_argument('--regressor-model', type=str, default='models/aesthetic_regressor.txt',
                       help='回归器模型路径')
    
    args = parser.parse_args()
    
    # 单个评估
    if args.template and args.copy:
        score = evaluate_calligraphy_aesthetic(
            template_path=args.template,
            copy_path=args.copy,
            srn_model_path=args.srn_model if os.path.exists(args.srn_model) else None,
            regressor_model_path=args.regressor_model if os.path.exists(args.regressor_model) else None
        )
        print(f"书法美学评分: {score:.2f}")
    
    # 批量评估
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
        else:
            batch_evaluate(
                input_file=args.input_file,
                output_file=args.output_file,
                srn_model_path=args.srn_model if os.path.exists(args.srn_model) else None,
                regressor_model_path=args.regressor_model if os.path.exists(args.regressor_model) else None
            )
    
    else:
        print("SRAFE书法美学评估系统")
        print("使用方法:")
        print("  单个评估: python main.py --template <模板路径> --copy <摹本路径>")
        print("  批量评估: python main.py --input-file <输入文件路径>")
