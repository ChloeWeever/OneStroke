"""
train.py - 完整训练流程

实现SRAFE系统的完整训练流程，包括SRN模型训练和回归器训练
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srn.train_srn import train_srn
from fusion.train_regressor import train_regressor
from code.utils.global_config import Config

def setup_logging(log_dir: str = 'logs') -> None:
    """
    设置日志记录
    
    Args:
        log_dir: 日志文件保存目录
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_config(config_path: str) -> Optional[Config]:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置对象，如果验证失败则返回None
        
    Raises:
        ValueError: 如果配置文件不存在或格式无效
    """
    try:
        if not os.path.exists(config_path):
            raise ValueError(f"配置文件不存在: {config_path}")
            
        config = Config(config_path)
        
        # 验证必需的配置项
        required_sections = ['data', 'srn', 'regressor']
        for section in required_sections:
            if section not in config.config:
                raise ValueError(f"配置文件缺少必需的{section}部分")
                
        return config
        
    except Exception as e:
        logging.error(f"配置验证失败: {str(e)}")
        return None

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='SRAFE训练脚本')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--train-srn', action='store_true',
                       help='训练SRN模型')
    parser.add_argument('--train-regressor', action='store_true',
                       help='训练回归器')
    parser.add_argument('--all', action='store_true',
                       help='训练所有模型')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 设置日志记录
    setup_logging()
    logging.info("开始SRAFE系统训练...")
    
    # 验证配置文件
    config = validate_config(args.config)
    if config is None:
        logging.error("配置验证失败，训练终止")
        return
    
    try:
        # 确保模型保存目录存在
        os.makedirs('models/srn', exist_ok=True)
        os.makedirs('models/regressor', exist_ok=True)
        
        # 训练SRN模型
        if args.train_srn or args.all:
            logging.info("开始训练SRN模型...")
            train_srn(args.config, resume=args.resume)
        
        # 训练回归器
        if args.train_regressor or args.all:
            logging.info("开始训练回归器...")
            train_regressor(args.config, resume=args.resume)
        
        logging.info("训练完成!")
        
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
