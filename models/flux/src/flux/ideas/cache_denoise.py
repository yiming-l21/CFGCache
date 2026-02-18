import torch
from ..model import Flux
from torch import Tensor
from ..modules.cache_functions import cache_init


def denoise_cache(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # cache mode
    cache_mode: str = "Taylor",
    # cache parameters
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    rel_l1_thresh: float = 0.6,
    # ClusCa parameters
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    # Analytic HiCache (HiCache-Analytic) parameters
    analytic_sigma_alpha: float | None = None,
    analytic_sigma_max: float | None = None,
    analytic_sigma_beta: float | None = None,
    analytic_sigma_eps: float | None = None,
    analytic_sigma_q_quantile: float | None = None,
    analytic_sigma_smooth: float | None = None,
    # 🔥 新增：特征收集参数
    enable_feature_collection: bool = False,
    feature_collection_config: dict = None,
):
    clusca_kwargs = None
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        clusca_kwargs = {
            "clusca_fresh_threshold": clusca_fresh_threshold,
            "clusca_cluster_num": clusca_cluster_num,
            "clusca_cluster_method": clusca_cluster_method,
            "clusca_k": clusca_k,
            "clusca_propagation_ratio": clusca_propagation_ratio,
        }
        clusca_kwargs = {k: v for k, v in clusca_kwargs.items() if v is not None}

    analytic_kwargs = None
    if cache_mode == "HiCache-Analytic":
        analytic_kwargs = {
            "analytic_sigma_alpha": analytic_sigma_alpha,
            "analytic_sigma_max": analytic_sigma_max,
            "analytic_sigma_beta": analytic_sigma_beta,
            "analytic_sigma_eps": analytic_sigma_eps,
            "analytic_sigma_q_quantile": analytic_sigma_q_quantile,
            "analytic_sigma_smooth": analytic_sigma_smooth,
        }
        analytic_kwargs = {k: v for k, v in analytic_kwargs.items() if v is not None}

    # init cache with specified mode and parameters
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        model_kwargs = clusca_kwargs
    elif cache_mode == "HiCache-Analytic":
        model_kwargs = analytic_kwargs
    elif cache_mode == "TeaCache":
        model_kwargs = {"rel_l1_thresh":rel_l1_thresh}
    else:
        model_kwargs = None
    cache_dic, current = cache_init(
        timesteps,
        model_kwargs=model_kwargs,
        mode=cache_mode,
        interval=interval,
        max_order=max_order,
        first_enhance=first_enhance,
        hicache_scale=hicache_scale,
    )

    # 🔥 新增：配置特征收集
    if enable_feature_collection:
        cache_dic["enable_feature_collection"] = True
        cache_dic["feature_collection_config"] = feature_collection_config or {
            "target_layer": 14,
            "target_module": "total",
            "target_stream": "single_stream",
        }

    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current["step"] = 0
    current["num_steps"] = len(timesteps) - 1
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        current["t"] = t_curr
        # print(t_curr)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic=cache_dic,
            current=current,
            guidance=guidance_vec,
        )
        # print(img.shape)
        img = img + (t_prev - t_curr) * pred
        current["step"] += 1

    # 🔥 新增：将cache_dic存储到模型中以便后续访问
    model._last_cache_dic = cache_dic

    return img

def denoise_cache_cfg(
    model: "Flux",
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor | None = None,
    neg_txt_ids: Tensor | None = None,
    neg_vec: Tensor | None = None,
    true_cfg_scale: float = 1.0,
    # sampling parameters
    timesteps: list[float] | None = None,
    guidance: float = 4.0,
    # cache mode
    cache_mode: str = "Taylor",
    # cache parameters
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    rel_l1_thresh: float = 0.6,
    # ClusCa parameters
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    # Analytic HiCache (HiCache-Analytic) parameters
    analytic_sigma_alpha: float | None = None,
    analytic_sigma_max: float | None = None,
    analytic_sigma_beta: float | None = None,
    analytic_sigma_eps: float | None = None,
    analytic_sigma_q_quantile: float | None = None,
    analytic_sigma_smooth: float | None = None,
    # 🔥 新增：特征收集参数
    enable_feature_collection: bool = False,
    feature_collection_config: dict = None,
):
    if timesteps is None:
        raise ValueError("timesteps must not be None")

    # True-CFG 开关：必须有 uncond 条件，且 scale>1 才做两次 forward
    do_true_cfg = (true_cfg_scale > 1.0) and (neg_txt is not None) and (neg_vec is not None)
    if do_true_cfg and neg_txt_ids is None:
        # 你也可以在外部保证传入；这里直接强约束，避免 silent bug
        raise ValueError("do_true_cfg=True but neg_txt_ids is None")

    # --------- build model_kwargs for cache_init ----------
    clusca_kwargs = None
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        clusca_kwargs = {
            "clusca_fresh_threshold": clusca_fresh_threshold,
            "clusca_cluster_num": clusca_cluster_num,
            "clusca_cluster_method": clusca_cluster_method,
            "clusca_k": clusca_k,
            "clusca_propagation_ratio": clusca_propagation_ratio,
        }
        clusca_kwargs = {k: v for k, v in clusca_kwargs.items() if v is not None}

    analytic_kwargs = None
    if cache_mode == "HiCache-Analytic":
        analytic_kwargs = {
            "analytic_sigma_alpha": analytic_sigma_alpha,
            "analytic_sigma_max": analytic_sigma_max,
            "analytic_sigma_beta": analytic_sigma_beta,
            "analytic_sigma_eps": analytic_sigma_eps,
            "analytic_sigma_q_quantile": analytic_sigma_q_quantile,
            "analytic_sigma_smooth": analytic_sigma_smooth,
        }
        analytic_kwargs = {k: v for k, v in analytic_kwargs.items() if v is not None}

    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        model_kwargs = clusca_kwargs
    elif cache_mode == "HiCache-Analytic":
        model_kwargs = analytic_kwargs
    elif cache_mode == "TeaCache":
        model_kwargs = {"rel_l1_thresh":rel_l1_thresh}
    else:
        model_kwargs = None

    # --------- init cache(s) ----------
    def _init_one_cache():
        cache_dic, current = cache_init(
            timesteps,
            model_kwargs=model_kwargs,
            mode=cache_mode,
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            hicache_scale=hicache_scale,
        )
        # feature collection：建议只挂在 cond 分支（否则会混两套统计）
        if enable_feature_collection:
            cache_dic["enable_feature_collection"] = True
            cache_dic["feature_collection_config"] = feature_collection_config or {
                "target_layer": 14,
                "target_module": "total",
                "target_stream": "single_stream",
            }
        current["step"] = 0
        current["num_steps"] = len(timesteps) - 1
        return cache_dic, current

    cache_dic_cond, current_cond = _init_one_cache()

    if do_true_cfg:
        # uncond 分支：默认不做 feature collection（避免你后处理不知取哪份）
        cache_dic_uncond, current_uncond = cache_init(
            timesteps,
            model_kwargs=model_kwargs,
            mode=cache_mode,
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            hicache_scale=hicache_scale,
        )
        current_uncond["step"] = 0
        current_uncond["num_steps"] = len(timesteps) - 1
    else:
        cache_dic_uncond, current_uncond = None, None

    # guidance_vec：你原来就传给 model，这里保持一致
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # --------- main sampling loop ----------
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # cond forward
        current_cond["t"] = t_curr
        pred_cond = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic=cache_dic_cond,
            current=current_cond,
            guidance=guidance_vec,
        )

        if do_true_cfg:
            # uncond forward (separate cache!)
            current_uncond["t"] = t_curr
            pred_uncond = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                cache_dic=cache_dic_uncond,
                current=current_uncond,
                guidance=guidance_vec,
            )
            # True CFG combine
            pred = pred_uncond + true_cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = pred_cond

        img = img + (t_prev - t_curr) * pred

        current_cond["step"] += 1
        if do_true_cfg:
            current_uncond["step"] += 1

    # 保存 cache 便于外部读取/调试
    if do_true_cfg:
        model._last_cache_dic = {"cond": cache_dic_cond, "uncond": cache_dic_uncond}
    else:
        model._last_cache_dic = cache_dic_cond

    return img

