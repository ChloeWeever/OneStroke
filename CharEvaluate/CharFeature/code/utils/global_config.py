"""
global_config.py - 配置管理

提供配置加载、验证和管理功能，支持YAML格式配置文件和环境变量。
支持配置项验证、异常处理和模块化配置。
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, field

class ConfigError(Exception):
    """配置相关异常的基类"""
    pass

class ConfigFileError(ConfigError):
    """配置文件加载异常"""
    pass

class ConfigValidationError(ConfigError):
    """配置验证异常"""
    pass

@dataclass
class ConfigValidator:
    """配置验证器基类"""
    def validate_path(self, path: str, should_exist: bool = True, 
                     is_file: bool = True, create_if_missing: bool = False) -> None:
        """验证路径配置"""
        try:
            p = Path(path)
            if should_exist:
                if is_file and not p.is_file():
                    raise ConfigValidationError(f"文件不存在或不是文件: {path}")
                if not is_file and not p.is_dir():
                    raise ConfigValidationError(f"目录不存在或不是目录: {path}")
            elif create_if_missing:
                if is_file:
                    p.parent.mkdir(parents=True, exist_ok=True)
                else:
                    p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if not isinstance(e, ConfigValidationError):
                raise ConfigValidationError(f"路径验证失败 {path}: {str(e)}")
    
    def validate_number(self, value: Union[int, float], min_value: Optional[Union[int, float]] = None,
                       max_value: Optional[Union[int, float]] = None, 
                       is_integer: bool = False) -> None:
        """验证数值配置"""
        try:
            if is_integer and not isinstance(value, int):
                raise ConfigValidationError(f"需要整数类型: {value}")
            if min_value is not None and value < min_value:
                raise ConfigValidationError(f"值小于最小值 {min_value}: {value}")
            if max_value is not None and value > max_value:
                raise ConfigValidationError(f"值大于最大值 {max_value}: {value}")
        except Exception as e:
            if not isinstance(e, ConfigValidationError):
                raise ConfigValidationError(f"数值验证失败: {str(e)}")
    
    def validate_enum(self, value: Any, valid_values: List[Any]) -> None:
        """验证枚举配置"""
        if value not in valid_values:
            raise ConfigValidationError(
                f"值 {value} 不在有效选项中: {valid_values}")

class Config:
    """配置管理类"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.validator = ConfigValidator()
        self.config_file = config_file
        self.config = {}
        self._env_prefix = "CALLIG_"  # 环境变量前缀
        
        if config_file:
            self.load_config(config_file)
            
        # 加载环境变量覆盖
        self._load_from_env()
        
        # 验证配置
        self._validate_config()
        self.config = {}
        
        if config_file:
            self.load_config(config_file)
    
    def _validate_config(self):
        """验证所有配置项"""
        try:
            # 验证路径类配置
            for path_key in ['data_dir', 'model_dir', 'output_dir']:
                if path_key in self.config:
                    self.validator.validate_path(
                        self.config[path_key], 
                        should_exist=True,
                        is_file=False
                    )
            
            # 验证训练相关配置
            if 'training' in self.config:
                train_cfg = self.config['training']
                if 'batch_size' in train_cfg:
                    self.validator.validate_number(
                        train_cfg['batch_size'],
                        min_value=1,
                        is_integer=True
                    )
                if 'epochs' in train_cfg:
                    self.validator.validate_number(
                        train_cfg['epochs'],
                        min_value=1,
                        is_integer=True
                    )
                if 'learning_rate' in train_cfg:
                    self.validator.validate_number(
                        train_cfg['learning_rate'],
                        min_value=0.0
                    )
                
            # 验证模型相关配置  
            if 'model' in self.config:
                model_cfg = self.config['model']
                if 'type' in model_cfg:
                    self.validator.validate_enum(
                        model_cfg['type'],
                        ['srn', 'faster_rcnn']
                    )
                    
        except ConfigValidationError as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            raise
            
    def _load_from_env(self):
        """从环境变量加载配置覆盖"""
        for env_key, env_value in os.environ.items():
            if env_key.startswith(self._env_prefix):
                # 移除前缀并转换为小写
                config_key = env_key[len(self._env_prefix):].lower()
                try:
                    # 尝试解析为JSON
                    value = json.loads(env_value)
                except json.JSONDecodeError:
                    # 不是JSON则保持原始字符串
                    value = env_value
                    
                # 更新配置
                self._update_nested_dict(self.config, config_key.split('_'), value)
                
    def _update_nested_dict(self, d: dict, key_parts: List[str], value: Any):
        """更新嵌套字典的辅助方法"""
        for i, part in enumerate(key_parts[:-1]):
            if part not in d:
                d[part] = {}
            elif not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[key_parts[-1]] = value

    def load_config(self, config_file: str):
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径
            
        Raises:
            ConfigFileError: 当配置文件不存在或无法解析时
        """
        try:
            if not os.path.exists(config_file):
                raise ConfigFileError(f"配置文件不存在: {config_file}")
                
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            if not isinstance(self.config, dict):
                raise ConfigFileError(f"配置文件格式错误: {config_file}")
                
        except yaml.YAMLError as e:
            raise ConfigFileError(f"配置文件解析失败 {config_file}: {str(e)}")
        except Exception as e:
            if not isinstance(e, ConfigFileError):
                raise ConfigFileError(f"加载配置文件失败 {config_file}: {str(e)}")
    
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键 (支持点号分隔的嵌套键，如 'srn.backbone')
            default: 默认值
            required: 是否为必需配置项
            
        Returns:
            value: 配置项值
            
        Raises:
            ConfigError: 当required=True且配置项不存在时
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if not isinstance(value, dict):
                    raise ConfigError(f"无法访问键路径 {key}: 不是嵌套字典")
                if k not in value:
                    if required:
                        raise ConfigError(f"缺少必需的配置项: {key}")
                    return default
                value = value[k]
            return value
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            if required:
                raise ConfigError(f"获取配置项失败 {key}: {str(e)}")
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置项
        
        Args:
            key: 配置项键 (支持点号分隔的嵌套键)
            value: 配置项值
        """
        # 支持点号分隔的嵌套键
        keys = key.split('.')
        config = self.config
        
        # 逐层创建嵌套字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置最终值
        config[keys[-1]] = value
    
    def save_config(self, config_file: str):
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路径
            
        Raises:
            ConfigFileError: 当保存配置文件失败时
        """
        try:
            # 验证配置有效性
            self._validate_config()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, allow_unicode=True)
                
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigFileError(f"保存配置文件失败 {config_file}: {str(e)}")
            
    def merge_config(self, other_config: Dict[str, Any]):
        """
        合并其他配置到当前配置
        
        Args:
            other_config: 要合并的配置字典
        """
        def _merge_dict(source: dict, update: dict):
            for key, value in update.items():
                if isinstance(value, dict) and key in source:
                    if isinstance(source[key], dict):
                        _merge_dict(source[key], value)
                    else:
                        source[key] = value
                else:
                    source[key] = value
                    
        try:
            if not isinstance(other_config, dict):
                raise ConfigError("合并配置失败: 输入必须是字典类型")
            _merge_dict(self.config, other_config)
            self._validate_config()
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"合并配置失败: {str(e)}")
            
    def reset_config(self):
        """
        重置配置为初始状态
        
        说明:
            重置配置字典为空，如果存在配置文件则重新加载
        """
        self.config = {}
        if self.config_file:
            self.load_config(self.config_file)

# 默认训练配置
DEFAULT_TRAINING_CONFIG = {
    'srn': {
        'backbone': 'resnet101',
        'pretrained': True,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50,
        'device': 'cuda'
    },
    'regressor': {
        'model_type': 'lightgbm',
        'params': {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100
        }
    },
    'data': {
        'data_dir': 'data',
        'train_list_file': 'data/metadata/train_list.csv',
        'val_list_file': 'data/metadata/val_list.csv',
        'test_list_file': 'data/metadata/test_list.csv'
    },
    'model_save_dir': 'models',
    'save_interval': 10,
    'num_workers': 4,
    'random_seed': 42
}

# 默认推理配置
DEFAULT_INFERENCE_CONFIG = {
    'models': {
        'srn_model_path': 'models/srn_resnet101.pth',
        'regressor_model_path': 'models/aesthetic_regressor.txt'
    },
    'device': 'cuda',
    'preprocessing': {
        'threshold': 127,
        'min_area': 50
    },
    'registration': {
        'target_size': 500
    },
    'features': {
        'num_handcrafted': 12,
        'num_deep': 10,
        'num_fused': 23
    }
}


'''
# 用于测试的示例代码
if __name__ == "__main__":
    # 创建默认训练配置文件
    train_config = Config()
    train_config.config = DEFAULT_TRAINING_CONFIG
    train_config.save_config('configs/training_config.yaml')
    
    # 创建默认推理配置文件
    infer_config = Config()
    infer_config.config = DEFAULT_INFERENCE_CONFIG
    infer_config.save_config('configs/inference_config.yaml')
    
    print("Default configuration files created successfully!")
'''