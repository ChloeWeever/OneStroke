"""
predict.py - 预测脚本

实现SRAFE系统的预测功能，支持单幅图像和批量预测模式
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.main import evaluate_calligraphy_aesthetic

def setup_logging(log_dir: str = 'logs/predict') -> None:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'predict.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_image_path(image_path: str, image_type: str) -> bool:
    """
    验证图像路径的有效性
    
    Args:
        image_path: 图像文件路径
        image_type: 图像类型描述("模板"或"摹本")
        
    Returns:
        is_valid: 路径是否有效
    """
    if not image_path:
        logging.error(f"{image_type}路径为空")
        return False
        
    path = Path(image_path)
    if not path.exists():
        logging.error(f"{image_type}文件不存在: {image_path}")
        return False
        
    if not path.is_file():
        logging.error(f"{image_type}路径不是文件: {image_path}")
        return False
        
    if path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
        logging.error(f"{image_type}文件格式无效: {image_path}")
        return False
        
    return True

def predict_single(template_path: str, copy_path: str,
                  srn_model_path: str = None,
                  regressor_model_path: str = None) -> Optional[float]:
    """
    预测单个书法作品的美学评分
    
    Args:
        template_path: 模板图像路径
        copy_path: 摹本图像路径
        srn_model_path: SRN模型路径 (可选)
        regressor_model_path: 回归器模型路径 (可选)
        
    Returns:
        score: 美学评分 (0-10)，如果预测失败则返回None
    """
    try:
        # 验证图像路径
        if not all([
            validate_image_path(template_path, "模板"),
            validate_image_path(copy_path, "摹本")
        ]):
            return None
            
        # 验证模型路径
        if srn_model_path and not Path(srn_model_path).is_file():
            logging.warning(f"SRN模型文件不存在: {srn_model_path}")
            srn_model_path = None
            
        if regressor_model_path and not Path(regressor_model_path).is_file():
            logging.warning(f"回归器模型文件不存在: {regressor_model_path}")
            regressor_model_path = None
            
        # 执行预测
        score = evaluate_calligraphy_aesthetic(
            template_path=template_path,
            copy_path=copy_path,
            srn_model_path=srn_model_path,
            regressor_model_path=regressor_model_path
        )
        
        # 验证预测结果
        if not isinstance(score, (int, float)):
            raise ValueError(f"预测结果类型无效: {type(score)}")
            
        if not 0 <= score <= 10:
            logging.warning(f"预测分数超出范围[0,10]: {score}")
            score = np.clip(score, 0, 10)
            
        return float(score)
        
    except Exception as e:
        logging.error(f"预测失败: {str(e)}")
        return None

def validate_input_file(input_file: str) -> Optional[pd.DataFrame]:
    """
    验证输入文件的有效性
    
    Args:
        input_file: CSV文件路径
        
    Returns:
        df: 加载的数据框，验证失败则返回None
    """
    try:
        # 检查文件存在性
        if not Path(input_file).is_file():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
            
        # 加载数据
        df = pd.read_csv(input_file)
        if len(df) == 0:
            raise ValueError("输入文件为空")
            
        # 检查必需的列
        required_columns = ['template_path', 'copy_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入文件缺少必需的列: {', '.join(missing_columns)}")
            
        return df
        
    except Exception as e:
        logging.error(f"输入文件验证失败: {str(e)}")
        return None

def predict_batch(input_file: str, output_file: str,
                 srn_model_path: str = None,
                 regressor_model_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    批量预测书法作品的美学评分
    
    Args:
        input_file: 输入文件路径 (CSV格式，包含template_path和copy_path列)
        output_file: 输出文件路径
        srn_model_path: SRN模型路径
        regressor_model_path: 回归器模型路径
        
    Returns:
        df: 预测结果数据框
        stats: 预测统计信息字典
    """
    # 验证输入文件
    df = validate_input_file(input_file)
    if df is None:
        raise ValueError("输入文件验证失败")
    
    # 初始化结果列
    df['predicted_score'] = np.nan
    df['prediction_status'] = ''
    
    total_samples = len(df)
    successful_predictions = 0
    failed_predictions = 0
    
    logging.info(f"开始预测 {total_samples} 个样本...")
    
    # 逐个预测
    for idx, row in df.iterrows():
        template_path = row['template_path']
        copy_path = row['copy_path']
        
        try:
            # 预测美学评分
            score = predict_single(
                template_path=template_path,
                copy_path=copy_path,
                srn_model_path=srn_model_path,
                regressor_model_path=regressor_model_path
            )
            
            if score is not None:
                df.loc[idx, 'predicted_score'] = score
                df.loc[idx, 'prediction_status'] = 'success'
                successful_predictions += 1
            else:
                df.loc[idx, 'prediction_status'] = 'failed'
                failed_predictions += 1
            
            # 定期报告进度
            if (idx + 1) % 100 == 0 or idx == total_samples - 1:
                progress = (idx + 1) / total_samples * 100
                logging.info(f"进度: {progress:.1f}% ({idx + 1}/{total_samples})")
                
        except Exception as e:
            df.loc[idx, 'prediction_status'] = 'error'
            df.loc[idx, 'error_message'] = str(e)
            failed_predictions += 1
            logging.error(f"样本 {idx} 预测失败: {str(e)}")
    
    # 计算统计信息
    stats = {
        'total_samples': total_samples,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'success_rate': successful_predictions / total_samples * 100 if total_samples > 0 else 0
    }
    
    if successful_predictions > 0:
        valid_scores = df['predicted_score'].dropna()
        stats.update({
            'mean_score': float(valid_scores.mean()),
            'std_score': float(valid_scores.std()),
            'min_score': float(valid_scores.min()),
            'max_score': float(valid_scores.max())
        })
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # 同时保存统计信息
    stats_file = output_path.with_stem(f"{output_path.stem}_stats").with_suffix('.csv')
    pd.DataFrame([stats]).to_csv(stats_file, index=False)
    
    logging.info(f"预测完成! 结果已保存到: {output_file}")
    logging.info(f"预测统计: 总样本={total_samples}, 成功={successful_predictions}, " + 
                f"失败={failed_predictions}, 成功率={stats['success_rate']:.1f}%")
    
    return df, stats

def main():
    """主预测函数"""
    parser = argparse.ArgumentParser(description='SRAFE书法美学评分预测系统')
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', action='store_true',
                           help='单个样本预测模式')
    mode_group.add_argument('--batch', action='store_true',
                           help='批量预测模式')
    
    parser.add_argument('--template', type=str, help='模板图像路径')
    parser.add_argument('--copy', type=str, help='摹本图像路径')
    parser.add_argument('--input-file', type=str, help='输入文件路径 (CSV格式)')
    parser.add_argument('--output-file', type=str, default='results/predictions.csv',
                       help='输出文件路径')
    parser.add_argument('--srn-model', type=str, default='models/srn_resnet101.pth',
                       help='SRN模型路径')
    parser.add_argument('--regressor-model', type=str, default='models/aesthetic_regressor.txt',
                       help='回归器模型路径')
    parser.add_argument('--log-dir', type=str, default='logs/predict',
                       help='日志保存目录')
    
    args = parser.parse_args()
    
    # 设置日志记录
    setup_logging(args.log_dir)
    logging.info("SRAFE预测系统启动")
    
    try:
        # 检查模型路径
        srn_model_path = Path(args.srn_model)
        regressor_model_path = Path(args.regressor_model)
        
        models_status = []
        if not srn_model_path.is_file():
            models_status.append(f"SRN模型不存在: {srn_model_path}")
            srn_model_path = None
        if not regressor_model_path.is_file():
            models_status.append(f"回归器模型不存在: {regressor_model_path}")
            regressor_model_path = None
            
        if models_status:
            logging.warning("模型文件状态:\n" + "\n".join(models_status))
        
        # 单个预测模式
        if args.single:
            if not all([args.template, args.copy]):
                logging.error("单个预测模式需要同时提供--template和--copy参数")
                return
                
            score = predict_single(
                template_path=args.template,
                copy_path=args.copy,
                srn_model_path=str(srn_model_path) if srn_model_path else None,
                regressor_model_path=str(regressor_model_path) if regressor_model_path else None
            )
            
            if score is not None:
                logging.info(f"预测完成!\n模板: {args.template}\n摹本: {args.copy}\n" + 
                           f"美学评分: {score:.2f}")
            else:
                logging.error("预测失败")
        
        # 批量预测模式
        elif args.batch:
            if not args.input_file:
                logging.error("批量预测模式需要提供--input-file参数")
                return
                
            try:
                df, stats = predict_batch(
                    input_file=args.input_file,
                    output_file=args.output_file,
                    srn_model_path=str(srn_model_path) if srn_model_path else None,
                    regressor_model_path=str(regressor_model_path) if regressor_model_path else None
                )
                
                logging.info("\n批量预测统计信息:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        logging.info(f"{key}: {value:.2f}")
                    else:
                        logging.info(f"{key}: {value}")
                        
            except Exception as e:
                logging.error(f"批量预测失败: {str(e)}")
                
    except Exception as e:
        logging.error(f"执行过程出错: {str(e)}")
        raise
        
    finally:
        logging.info("SRAFE预测系统关闭")

if __name__ == "__main__":
    main()
