import numpy as np
def cache_init(
    timesteps,
    model_kwargs=None,
    mode="Taylor",
    interval=None,
    max_order=None,
    first_enhance=None,
    hicache_scale=None,
):
    """
    Initialization for cache.

    :param timesteps: 时间步序列
    :param model_kwargs: 模型参数（暂未使用）
    :param mode: 缓存模式 ('original', 'ToCa', 'Taylor', 'Taylor-Scaled', 'HiCache', 'HiCache-Analytic', 'Delta', 'collect', 'ClusCa', 'Hi-ClusCa')
    :param interval: 缓存刷新间隔 (覆盖默认值)
    :param max_order: 最大阶数 (覆盖默认值)
    :param first_enhance: 初始增强步数 (覆盖默认值)
    :param hicache_scale: HiCache多项式缩放因子 (覆盖默认值)
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}
    cache_dic["attn_map"] = {}
    cache_dic["attn_map"][-1] = {}
    cache_dic["attn_map"][-1]["double_stream"] = {}
    cache_dic["attn_map"][-1]["single_stream"] = {}

    cache_dic["k-norm"] = {}
    cache_dic["k-norm"][-1] = {}
    cache_dic["k-norm"][-1]["double_stream"] = {}
    cache_dic["k-norm"][-1]["single_stream"] = {}

    cache_dic["v-norm"] = {}
    cache_dic["v-norm"][-1] = {}
    cache_dic["v-norm"][-1]["double_stream"] = {}
    cache_dic["v-norm"][-1]["single_stream"] = {}

    cache_dic["cross_attn_map"] = {}
    cache_dic["cross_attn_map"][-1] = {}
    cache[-1]["double_stream"] = {}
    cache[-1]["single_stream"] = {}
    cache_dic["cache_counter"] = 0

    for j in range(19):
        cache[-1]["double_stream"][j] = {}
        cache_index[-1][j] = {}
        cache_dic["attn_map"][-1]["double_stream"][j] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["total"] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["img_mlp"] = {}

        cache_dic["k-norm"][-1]["double_stream"][j] = {}
        cache_dic["k-norm"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["k-norm"][-1]["double_stream"][j]["img_mlp"] = {}

        cache_dic["v-norm"][-1]["double_stream"][j] = {}
        cache_dic["v-norm"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["v-norm"][-1]["double_stream"][j]["img_mlp"] = {}

    for j in range(38):
        cache[-1]["single_stream"][j + 19] = {}  # 单流块缓存也使用全局层索引
        cache_index[-1][j + 19] = {}  # 单流块从层19开始编号
        cache_dic["attn_map"][-1]["single_stream"][j] = {}
        cache_dic["attn_map"][-1]["single_stream"][j]["total"] = {}

        cache_dic["k-norm"][-1]["single_stream"][j] = {}
        cache_dic["k-norm"][-1]["single_stream"][j]["total"] = {}

        cache_dic["v-norm"][-1]["single_stream"][j] = {}
        cache_dic["v-norm"][-1]["single_stream"][j]["total"] = {}

    cache_dic["taylor_cache"] = False
    cache_dic["Delta-DiT"] = False
    cache_dic["use_hicache"] = False
    cache_dic["mode"] = mode

    if mode == "original":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3
    elif mode == "FasterCache":
        cache_dic["cache_type"] = "random"       
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "FasterCache"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

        cache_dic.setdefault("fastercache", {})
        cache_dic["fastercache"].setdefault("cfg", {})
        fc = cache_dic["fastercache"]["cfg"]
        fc.setdefault("enabled", False)
        fc["warmup"] = int(cache_dic.get("first_enhance", 0))
        fc["cfg_mode"] = "fft"
        fc["refresh_period"] = 5          # paper implementation
        fc["alpha1"] = 0.2                # paper implementation
        fc["alpha2"] = 0.2                # paper implementation
        fc["t0_ratio"] = 0.5              
        fc.setdefault("delta", None)      
        fc.setdefault("has_delta", False)
        fc.setdefault("delta_lf", None)
        fc.setdefault("delta_hf", None)
        fc.setdefault("has_delta_fft", False)
        fc.setdefault("cnt", 0)     

        # -------- DFR (dynamic feature reuse) --------
        cache_dic["fastercache"].setdefault("reuse", {})
        fr = cache_dic["fastercache"]["reuse"]
        fr.setdefault("enabled", True)
        fr["warmup"] = fc["warmup"]
        fr["refresh_period"] = 2  # alternate timesteps
        fr.setdefault("cnt", 0)
        fr.setdefault("attn_prev", {})    # dict[(stream, layer)] = Tensor
        fr.setdefault("attn_prev2", {})   # dict[(stream, layer)] = Tensor
        fr.setdefault("has_prev", set())  # set of keys that have prev
        fr.setdefault("has_prev2", set())
    elif mode == "TeaCache":
        cache_dic["cache_type"] = "random"        
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache

        cache_dic["fresh_ratio_schedule"] = "TeaCache"
        cache_dic["fresh_ratio"] = 0.0          
        cache_dic["fresh_threshold"] = 1         
        cache_dic["cal_threshold"] = 1          
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3           

        model_kwargs = model_kwargs or {}
        cache_dic["teacache_enable"] = True
        cache_dic["teacache"] = {
            "cnt": 0,
            "num_steps": len(timesteps),   
            "rel_l1_thresh": model_kwargs.get("rel_l1_thresh", 0.8),
            "accumulated_rel_l1_distance": 0.0,
            "previous_modulated_input": None,   
            "previous_residual": None,          
            "coefficients": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
        }
    elif mode == "MagCache":
        def nearest_interp(src_array, target_length):
            src_length = len(src_array)
            if target_length == 1:
                return np.array([src_array[-1]])
            scale = (src_length - 1) / (target_length - 1)
            mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
            return src_array[mapped_indices]
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "MagCache"
        cache_dic["fresh_ratio"] = 0.0          
        cache_dic["fresh_threshold"] = 1         
        cache_dic["cal_threshold"] = 1          
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3 

        model_kwargs = model_kwargs or {}
        base_mag_ratios = np.array([
            1.00000, 1.15589, 1.03062, 1.06443, 1.05367, 1.03530, 1.03932, 1.02461, 1.02777,
            1.01864, 1.02197, 1.01830, 1.01566, 1.01181, 1.01239, 1.00975, 1.01173, 1.00840,
            1.00098, 1.00147, 1.00974, 1.00173, 1.01344, 1.00718, 0.99618, 1.00715, 1.00924,
            1.00119, 1.00006, 1.00295, 0.99880, 1.01076, 0.99010, 1.00541, 1.00079, 0.99685,
            0.99504, 0.99612, 0.98827, 0.99784, 0.99364, 0.99299, 0.98553, 0.98508, 0.98461,
            0.96980, 0.96539, 0.95062, 0.92728, 0.92891,
        ], dtype=np.float64)

        ratios_in = model_kwargs.get("mag_ratios", base_mag_ratios)
        ratios_in = np.asarray(ratios_in, dtype=np.float64)
        T = len(timesteps)  # num_steps
        if len(ratios_in) != T:
            ratios_aligned = nearest_interp(ratios_in, T)
        else:
            ratios_aligned = ratios_in
        mc = {
            "cnt": 0,
            "num_steps": len(timesteps),

            # === required hyperparams (same names as diffusers demo) ===
            "K": int(model_kwargs.get("K", 5)),
            "magcache_thresh": float(model_kwargs.get("magcache_thresh", 0.24)),
            "retention_ratio": float(model_kwargs.get("retention_ratio", 0.1)),

            # === required state ===
            "accumulated_ratio": 1.0,
            "accumulated_err": 0.0,
            "accumulated_steps": 0,

            # === calibrated ratios ===
            "mag_ratios": ratios_aligned,

            # === residual cache ===
            "previous_residual": None,
        }
        cache_dic["magcache"] = mc
        cache_dic["magcache_enable"] = True
    elif mode == "ToCa":
        cache_dic["cache_type"] = "attention"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 2
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor-Scaled":
        # Taylor + 双重缩放（用于消融对比）
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 2
        cache_dic["first_enhance"] = 3
        cache_dic["prediction_mode"] = "taylor_scaled"
        # 允许通过 model_kwargs 或外部参数传入缩放因子
        model_kwargs = model_kwargs or {}
        cache_dic["hicache_scale_factor"] = model_kwargs.get(
            "hicache_scale_factor", model_kwargs.get("hicache_scale", 0.5)
        )

    elif mode == "HiCache":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["use_hicache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        cache_dic["hicache_scale_factor"] = 0.5  # HiCache 多项式缩放因子

    elif mode == "HiCache-Analytic":
        # Analytic Adaptive Sigma version of HiCache
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["use_hicache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        
        # Enable Analytic Sigma
        cache_dic["analytic_sigma"] = True
        cache_dic["analytic_stats"] = {} # Initialize stats storage
        
        # Config
        model_kwargs = model_kwargs or {}
        cache_dic["analytic_sigma_config"] = {
            # Default alpha≈1.28 gives initial sigma≈0.9 when q≈1 (raw = alpha / sqrt(2))
            "alpha": model_kwargs.get("analytic_sigma_alpha", 1.28),
            "sigma_max": model_kwargs.get("analytic_sigma_max", 1.0),
            "beta": model_kwargs.get("analytic_sigma_beta", 0.01),
            "eps": model_kwargs.get("analytic_sigma_eps", 1e-6),
            # 选填：更稳的统计与更平滑的 sigma
            "q_quantile": model_kwargs.get("analytic_sigma_q_quantile"),  # e.g. 0.95
            "sigma_smooth": model_kwargs.get("analytic_sigma_smooth", 0.0),  # gamma in log sigma EMA
        }

    elif mode == "Delta":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 3
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["Delta-DiT"] = True
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1

    elif mode == "collect":
        # 🔥 新增：专门用于特征收集的模式
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "collect"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1  # 每步都刷新
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = False  # 不使用Taylor缓存
        cache_dic["max_order"] = 0  # 不需要高阶展开
        cache_dic["first_enhance"] = len(timesteps)  # 所有步骤都是enhanced
        cache_dic["collect_mode"] = True  # 标记为收集模式
        # 自动启用特征收集
        cache_dic["enable_feature_collection"] = True

    elif mode == "ClusCa":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        cache_dic["cluster_info"] = {}
        cache_dic["cluster_num"] = 16
        cache_dic["cluster_method"] = "kmeans"
        cache_dic["k"] = 1
        cache_dic["propagation_ratio"] = 0.005
        cache_dic["prediction_mode"] = "taylor"  # ClusCa uses Taylor predictor

        cluster_info_dict = {}
        cluster_info_dict["cluster_indices"] = None
        cluster_info_dict["centroids"] = None

        cache_dic["cluster_info"]["double_stream"] = {}
        cache_dic["cluster_info"]["single_stream"] = {}
        cache_dic["cluster_info"]["double_stream"]["img_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["double_stream"]["txt_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["single_stream"]["total"] = cluster_info_dict.copy()

        # Allow override from model_kwargs for ClusCa
        model_kwargs = model_kwargs or {}
        if "clusca_fresh_threshold" in model_kwargs:
            cache_dic["fresh_threshold"] = model_kwargs["clusca_fresh_threshold"]
        if "clusca_cluster_num" in model_kwargs:
            cache_dic["cluster_num"] = model_kwargs["clusca_cluster_num"]
        if "clusca_cluster_method" in model_kwargs:
            cache_dic["cluster_method"] = model_kwargs["clusca_cluster_method"]
        if "clusca_k" in model_kwargs:
            cache_dic["k"] = model_kwargs["clusca_k"]
        if "clusca_propagation_ratio" in model_kwargs:
            cache_dic["propagation_ratio"] = model_kwargs["clusca_propagation_ratio"]

    elif mode == "Hi-ClusCa":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        cache_dic["cluster_info"] = {}
        cache_dic["cluster_num"] = 16
        cache_dic["cluster_method"] = "kmeans"
        cache_dic["k"] = 1
        cache_dic["propagation_ratio"] = 0.005
        cache_dic["prediction_mode"] = "hicache"  # Hi-ClusCa uses HiCache predictor

        cluster_info_dict = {}
        cluster_info_dict["cluster_indices"] = None
        cluster_info_dict["centroids"] = None

        cache_dic["cluster_info"]["double_stream"] = {}
        cache_dic["cluster_info"]["single_stream"] = {}
        cache_dic["cluster_info"]["double_stream"]["img_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["double_stream"]["txt_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["single_stream"]["total"] = cluster_info_dict.copy()

        # Allow override from model_kwargs for Hi-ClusCa
        model_kwargs = model_kwargs or {}
        if "clusca_fresh_threshold" in model_kwargs:
            cache_dic["fresh_threshold"] = model_kwargs["clusca_fresh_threshold"]
        if "clusca_cluster_num" in model_kwargs:
            cache_dic["cluster_num"] = model_kwargs["clusca_cluster_num"]
        if "clusca_cluster_method" in model_kwargs:
            cache_dic["cluster_method"] = model_kwargs["clusca_cluster_method"]
        if "clusca_k" in model_kwargs:
            cache_dic["k"] = model_kwargs["clusca_k"]
        if "clusca_propagation_ratio" in model_kwargs:
            cache_dic["propagation_ratio"] = model_kwargs["clusca_propagation_ratio"]
        if "hicache_scale" in model_kwargs:
            cache_dic["hicache_scale"] = model_kwargs["hicache_scale"]
        cache_dic["cluster_info"]["single_stream"]["total"] = cluster_info_dict.copy()

    current = {}
    current["final_time"] = timesteps[-2]
    current["activated_steps"] = []

    # Override default values with provided parameters
    if interval is not None:
        cache_dic["fresh_threshold"] = interval
    if max_order is not None:
        cache_dic["max_order"] = max_order
    if first_enhance is not None:
        cache_dic["first_enhance"] = first_enhance
    if hicache_scale is not None:
        cache_dic["hicache_scale_factor"] = hicache_scale

    return cache_dic, current
