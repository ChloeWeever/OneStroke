"""
evaluate.py - 评估脚本

实现SRAFE系统的评估功能，支持模型性能评估和结果分析
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.main import evaluate_calligraphy_aesthetic
from code.evaluation.metrics import calculate_mae, calculate_pcc
from code.utils.global_config import Config

def setup_logging(log_dir: str = 'logs/evaluate') -> None:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'evaluate_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_test_file(test_list_file: str) -> Optional[pd.DataFrame]:
    """
    验证测试集文件的有效性
    
    Args:
        test_list_file: 测试集文件路径
        
    Returns:
        df: 验证通过的数据框，失败则返回None
    """
    try:
        if not Path(test_list_file).is_file():
            raise FileNotFoundError(f"测试集文件不存在: {test_list_file}")
            
        df = pd.read_csv(test_list_file)
        if len(df) == 0:
            raise ValueError("测试集为空")
            
        required_columns = ['template_path', 'copy_path', 'score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"测试集缺少必需的列: {', '.join(missing_columns)}")
            
        return df
        
    except Exception as e:
        logging.error(f"测试集验证失败: {str(e)}")
        return None

def evaluate_model(test_list_file: str, 
                  srn_model_path: str = None,
                  regressor_model_path: str = None,
                  save_details: bool = True) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        test_list_file: 测试集列表文件路径
        srn_model_path: SRN模型路径
        regressor_model_path: 回归器模型路径
        save_details: 是否保存详细评估结果
        
    Returns:
        metrics: 评估指标字典，包含:
            - mae: 平均绝对误差
            - pcc: 皮尔逊相关系数
            - num_samples: 评估样本数
            - failed_samples: 失败样本数
            - evaluation_time: 评估耗时(秒)
            - error_analysis: 误差分析结果
    """
    start_time = time.time()
    
    # 验证测试集
    test_df = validate_test_file(test_list_file)
    if test_df is None:
        return {}
        
    # 初始化结果存储
    results_df = test_df.copy()
    results_df['predicted_score'] = np.nan
    results_df['error'] = np.nan
    results_df['evaluation_status'] = ''
    
    total_samples = len(test_df)
    successful_evaluations = 0
    failed_evaluations = 0
    
    logging.info(f"开始评估 {total_samples} 个样本...")
    
    # 逐个评估样本
    for idx, row in test_df.iterrows():
        template_path = row['template_path']
        copy_path = row['copy_path']
        true_score = row['score']
        
        try:
            # 验证文件路径
            if not (Path(template_path).is_file() and Path(copy_path).is_file()):
                raise FileNotFoundError(f"图像文件不存在")
            
            # 评估美学评分
            pred_score = evaluate_calligraphy_aesthetic(
                template_path=template_path,
                copy_path=copy_path,
                srn_model_path=srn_model_path,
                regressor_model_path=regressor_model_path
            )
            
            # 验证评分
            if not isinstance(pred_score, (int, float)) or not (0 <= pred_score <= 10):
                raise ValueError(f"无效的预测分数: {pred_score}")
                
            # 记录结果
            results_df.loc[idx, 'predicted_score'] = pred_score
            results_df.loc[idx, 'error'] = abs(pred_score - true_score)
            results_df.loc[idx, 'evaluation_status'] = 'success'
            successful_evaluations += 1
            
            # 定期报告进度
            if (idx + 1) % 100 == 0:
                progress = (idx + 1) / total_samples * 100
                logging.info(f"进度: {progress:.1f}% ({idx + 1}/{total_samples})")
                
        except Exception as e:
            results_df.loc[idx, 'evaluation_status'] = 'failed'
            results_df.loc[idx, 'error_message'] = str(e)
            failed_evaluations += 1
            logging.error(f"样本评估失败 (template={template_path}, copy={copy_path}): {str(e)}")
    
    # 计算评估指标
    evaluation_time = time.time() - start_time
    metrics = {}
    
    if successful_evaluations > 0:
        # 获取成功评估的样本
        valid_results = results_df[results_df['evaluation_status'] == 'success']
        predictions = valid_results['predicted_score'].values
        labels = valid_results['score'].values
        
        # 计算基本指标
        mae = calculate_mae(labels, predictions)
        pcc, p_value = calculate_pcc(labels, predictions)
        
        # 误差分析
        errors = np.abs(predictions - labels)
        error_analysis = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'median_error': float(np.median(errors)),
            'error_quartiles': [float(np.percentile(errors, p)) for p in [25, 50, 75]],
        }
        
        # 统计过大/过小预测的比例
        overpredictions = np.sum(predictions > labels) / len(predictions)
        underpredictions = np.sum(predictions < labels) / len(predictions)
        
        metrics = {
            'mae': mae,
            'pcc': pcc,
            'p_value': p_value,
            'num_samples': total_samples,
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'success_rate': successful_evaluations / total_samples * 100,
            'evaluation_time': evaluation_time,
            'error_analysis': error_analysis,
            'prediction_bias': {
                'overprediction_rate': float(overpredictions),
                'underprediction_rate': float(underpredictions)
            }
        }
        
        # 记录评估结果
        logging.info("\n评估完成!")
        logging.info(f"样本总数: {total_samples}")
        logging.info(f"成功评估: {successful_evaluations}")
        logging.info(f"评估失败: {failed_evaluations}")
        logging.info(f"成功率: {metrics['success_rate']:.1f}%")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"PCC: {pcc:.4f} (p={p_value:.4e})")
        logging.info(f"评估耗时: {evaluation_time:.2f}秒")
        
        # 保存详细结果
        if save_details:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = Path('results/evaluation')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存评估指标
            metrics_file = results_dir / f'metrics_{timestamp}.json'
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            # 保存详细结果
            results_file = results_dir / f'details_{timestamp}.csv'
            results_df.to_csv(results_file, index=False)
            
            logging.info(f"详细结果已保存到: {results_dir}")
    
    else:
        logging.error("没有成功评估的样本")
        metrics = {
            'num_samples': total_samples,
            'successful_evaluations': 0,
            'failed_evaluations': failed_evaluations,
            'success_rate': 0.0,
            'evaluation_time': evaluation_time
        }
    
    return metrics

def validate_paths(test_list_file: str, srn_model_path: str, 
                   regressor_model_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    验证文件路径
    
    Args:
        test_list_file: 测试集文件路径
        srn_model_path: SRN模型路径
        regressor_model_path: 回归器模型路径
        
    Returns:
        验证后的路径元组 (test_list_file, srn_model_path, regressor_model_path)
        
    Raises:
        ValueError: 如果必需的文件不存在
    """
    # 验证测试集文件
    if not Path(test_list_file).is_file():
        raise ValueError(f"测试集文件不存在: {test_list_file}")
    
    # 验证模型文件
    srn_path = Path(srn_model_path)
    regressor_path = Path(regressor_model_path)
    
    if not srn_path.is_file():
        logging.warning(f"SRN模型文件不存在: {srn_path}")
        srn_model_path = None
        
    if not regressor_path.is_file():
        logging.warning(f"回归器模型文件不存在: {regressor_path}")
        regressor_model_path = None
        
    return test_list_file, srn_model_path, regressor_model_path

def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='SRAFE评估系统')
    parser.add_argument('--test-list', type=str, default='data/metadata/test_list.csv',
                       help='测试集列表文件路径')
    parser.add_argument('--srn-model', type=str, default='models/srn_resnet101.pth',
                       help='SRN模型路径')
    parser.add_argument('--regressor-model', type=str, default='models/aesthetic_regressor.txt',
                       help='回归器模型路径')
    parser.add_argument('--save-details', action='store_true',
                       help='保存详细评估结果')
    parser.add_argument('--log-dir', type=str, default='logs/evaluate',
                       help='日志保存目录')
    
    args = parser.parse_args()
    
    # 设置日志记录
    setup_logging(args.log_dir)
    logging.info("SRAFE评估系统启动")
    
    try:
        # 验证文件路径
        test_list_file, srn_model_path, regressor_model_path = validate_paths(
            args.test_list, args.srn_model, args.regressor_model)
        
        # 执行评估
        metrics = evaluate_model(
            test_list_file=test_list_file,
            srn_model_path=srn_model_path,
            regressor_model_path=regressor_model_path,
            save_details=args.save_details
        )
        
        # 分析评估结果
        if metrics:
            if metrics.get('successful_evaluations', 0) > 0:
                # 记录主要指标
                logging.info("\n主要评估指标:")
                logging.info(f"MAE: {metrics['mae']:.4f}")
                logging.info(f"PCC: {metrics['pcc']:.4f} (p={metrics['p_value']:.4e})")
                
                # 记录错误分析
                error_analysis = metrics.get('error_analysis', {})
                if error_analysis:
                    logging.info("\n错误分析:")
                    logging.info(f"平均误差: {error_analysis['mean_error']:.4f}")
                    logging.info(f"最大误差: {error_analysis['max_error']:.4f}")
                    logging.info(f"最小误差: {error_analysis['min_error']:.4f}")
                    logging.info(f"中位误差: {error_analysis['median_error']:.4f}")
                    
                # 记录预测偏差
                bias = metrics.get('prediction_bias', {})
                if bias:
                    logging.info("\n预测偏差分析:")
                    logging.info(f"过高预测比例: {bias['overprediction_rate']:.2%}")
                    logging.info(f"过低预测比例: {bias['underprediction_rate']:.2%}")
            else:
                logging.error("评估失败: 没有成功评估的样本")
                
    except Exception as e:
        logging.error(f"评估过程出错: {str(e)}")
        raise
        
    finally:
        logging.info("SRAFE评估系统关闭")

if __name__ == "__main__":
    main()
