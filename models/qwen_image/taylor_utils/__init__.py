from typing import Dict
import torch
import math

from models.hicache_fast_impl import (
    derivative_approximation as hicache_derivative,
    taylor_formula as hicache_formula,
    taylor_cache_init as hicache_cache_init,
)

# 导入分组TaylorSeer功能
from .grouped_taylor import (
    DimensionGroupedTaylorSeer,
    create_grouped_taylor_cache,
    grouped_derivative_approximation,
    grouped_taylor_formula,
)


def _ensure_hicache_keys(current: Dict):
    current["stream"] = current.get("model", current.get("stream", "cond"))
    current["module"] = current.get("block", current.get("module", "img_attn"))

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if cache_dic.get('prediction_mode') == 'hicache':
        _ensure_hicache_keys(current)
        hicache_derivative(cache_dic, current, feature)
        return

    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['model']][current['layer']][current['block']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['model']][current['layer']][current['block']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = updated_taylor_factors

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if cache_dic.get('prediction_mode') == 'hicache':
        _ensure_hicache_keys(current)
        return hicache_formula(cache_dic, current)

    x = current['step'] - current['activated_steps'][-1]
    #x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic['cache'][-1][current['model']][current['layer']][current['block']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['model']][current['layer']][current['block']][i] * (x ** i)
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if cache_dic.get('prediction_mode') == 'hicache':
        _ensure_hicache_keys(current)
        hicache_cache_init(cache_dic, current)
        return

    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = {}


def grouped_taylor_cache_init(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Initialize grouped Taylor cache for dimension-wise TaylorSeer.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Current feature tensor to determine dimensions
    """
    if cache_dic.get('use_grouped_taylor', False):
        feature_dim = feature.shape[-1]
        create_grouped_taylor_cache(cache_dic, current, feature_dim)

def derivative_approximation_foca(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    缓存函数值序列：F_n, F_{n-1}, ...
    """
    cache_ref = cache_dic['cache'][-1][current['model']][current['layer']][current['block']]

    if not isinstance(cache_ref, list):
        cache_ref = []

    cache_ref.insert(0, feature.detach())

    max_keep = cache_dic.get('max_order', 2) + 1
    cache_ref = cache_ref[:max_keep]

    cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = cache_ref


def taylor_formula_foca(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    使用真实函数值序列，实现 BDF-2 + Heun 二阶预测器。
    :param cache_dic: 缓存字典
    :param current: 当前层/步的信息
    :return: 预测特征张量
    """
    feats_list = cache_dic['cache'][-1][current['model']][current['layer']][current['block']]

    if len(feats_list) < 2:
        return feats_list[0]  # 不足两帧，直接返回当前值

    # 获取历史值：F_n 和 F_{n-1}
    F_n = feats_list[0]
    F_nm1 = feats_list[1]

    # Step 1: BDF2 预测
    F_pred = (4.0 * F_n - F_nm1) / 3.0

    # Step 2: Heun 修正
    F_out = 0.5 * (F_n + F_pred)

    return F_out


def taylor_cache_init_foca(cache_dic: Dict, current: Dict):
    """
    初始化缓存为值序列 list 而不是导数 dict。
    """
    if (current['step'] == 0) and cache_dic.get('taylor_cache', True):
        cache_dic['cache'][-1][current['model']][current['layer']][current['block']] = []

def pipeline_with_taylorseer(pipe):
    """
    Pipeline with Taylorseer.
    :param pipe: QwenImagePipeline
    """
    import types
    from ..transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel
    from ..transformer_qwenimage import QwenImageTransformerBlock as LocalQwenImageTransformerBlock

    # Replace the transformer's forward method
    pipe.transformer.forward = types.MethodType(LocalQwenImageTransformer2DModel.forward, pipe.transformer)
    
    # Replace each transformer block's forward method and attention processor
    for _, block in enumerate(pipe.transformer.transformer_blocks):
        block.forward = types.MethodType(LocalQwenImageTransformerBlock.forward, block)

    return pipe
