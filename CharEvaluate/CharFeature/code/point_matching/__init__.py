"""
point_matching模块的初始化文件，导出公共接口
"""

from .cdp import CDPMatcher, match_keypoints
from ..utils.point_matching_utils import (
    convert_rcnn_to_points,
    visualize_point_matches,
    calculate_registration_error,
    remove_outliers
)

__all__ = [
    'CDPMatcher',
    'match_keypoints',
    'convert_rcnn_to_points',
    'visualize_point_matches',
    'calculate_registration_error',
    'remove_outliers'
]

# 版本信息
__version__ = '1.0.0'

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = ['numpy', 'opencv-python', 'pyyaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}"
        )

# 导入时检查依赖
check_dependencies()
