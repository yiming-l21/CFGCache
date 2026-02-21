from .force_scheduler import force_scheduler
import numpy as np

def cal_type(cache_dic, current):
    """
    Determine calculation type for this step
    """
    # 🔥 添加调试信息：在step 0-2时打印配置信息
    # if current['step'] <= 2:
    #     debug_info = (
    #         f"[CACHE DEBUG] Step {current['step']}: "
    #         f"fresh_ratio={cache_dic['fresh_ratio']}, "
    #         f"taylor_cache={cache_dic['taylor_cache']}, "
    #         f"fresh_threshold={cache_dic['fresh_threshold']}, "
    #         f"first_enhance={cache_dic.get('first_enhance', 'N/A')}"
    #     )
    #     print(debug_info)

    # 🔥 新增：collect 模式 - 专门用于特征收集
    if cache_dic.get("collect_mode", False):
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 COLLECT 模式 -> type='full' (特征收集)")
        return

    # 硬性守卫：original 模式无论 interval/first_enhance 如何设置，都应每步 full
    if cache_dic.get("mode") == "original":
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        return
    # =========================
    # TeaCache branch (paper-faithful)
    # =========================
    if cache_dic.get("mode") == "TeaCache":
        tc = cache_dic.get("teacache", None)
        if tc is None or (not cache_dic.get("teacache_enable", True)):
            # fallback: always full
            current["type"] = "full"
            if current["step"] == 0:
                current["activated_steps"].append(current["step"])
            return

        step = current["step"]
        T = int(tc.get("num_steps", current.get("num_steps", 0) or 0))

        # This must be provided by model forward BEFORE calling cal_type
        mod_inp = current.get("teacache_modulated_inp", None)

        # default: full
        current["type"] = "full"

        # cnt-based hard guard (paper): first & last step always full
        cnt = int(tc.get("cnt", 0))
        if cnt == 0 or (T > 0 and cnt == T - 1):
            tc["accumulated_rel_l1_distance"] = 0.0
            tc["previous_modulated_input"] = mod_inp
            current["type"] = "full"
            current["activated_steps"].append(step)

            # update counter (paper)
            tc["cnt"] = cnt + 1
            if T > 0 and tc["cnt"] == T:
                tc["cnt"] = 0
            return

        # If we cannot compute decision -> full
        prev = tc.get("previous_modulated_input", None)
        if mod_inp is None or prev is None:
            tc["previous_modulated_input"] = mod_inp
            current["type"] = "full"
            current["activated_steps"].append(step)

            # update counter
            tc["cnt"] = cnt + 1
            if T > 0 and tc["cnt"] == T:
                tc["cnt"] = 0
            return

        # rel l1 distance (paper): mean(|x_t - x_{t-1}|) / mean(|x_{t-1}|)
        eps = 1e-6
        denom = prev.abs().mean() + eps
        rel = (mod_inp - prev).abs().mean() / denom
        rel = float(rel.detach().cpu())

        # polynomial rescale (paper coefficients)
        coeff = tc.get("coefficients", None)
        if coeff is not None:
            # 4th order poly: a x^4 + b x^3 + c x^2 + d x + e
            # computed as (((a*x + b)*x + c)*x + d)*x + e
            rel_scaled = (((coeff[0] * rel + coeff[1]) * rel + coeff[2]) * rel + coeff[3]) * rel + coeff[4]
        else:
            rel_scaled = rel

        tc["accumulated_rel_l1_distance"] = float(tc.get("accumulated_rel_l1_distance", 0.0)) + float(rel_scaled)

        # decision
        if tc["accumulated_rel_l1_distance"] < float(tc.get("rel_l1_thresh", 0.6)):
            current["type"] = "TeaCacheSkip"
            # do NOT append activated_steps when skipping
        else:
            current["type"] = "full"
            tc["accumulated_rel_l1_distance"] = 0.0
            current["activated_steps"].append(step)

        # update prev input and counter (paper)
        tc["previous_modulated_input"] = mod_inp
        tc["cnt"] = cnt + 1
        if T > 0 and tc["cnt"] == T:
            tc["cnt"] = 0
        return
    if cache_dic.get("mode") == "MagCache":
        mc = cache_dic["magcache"]
        step = int(current.get("step", 0))
        T = int(mc["num_steps"])
        current["type"] = "full"

        if mc.get("previous_residual", None) is None:
            current.setdefault("activated_steps", []).append(step)
            mc["cnt"] = (mc["cnt"] + 1) % max(1, T)
            return

        skip_forward = False
        if mc["cnt"] >= int(mc["retention_ratio"] * mc["num_steps"] + 0.5):
            cur_scale = float(mc["mag_ratios"][mc["cnt"]])
            mc["accumulated_ratio"] *= cur_scale
            mc["accumulated_steps"] += 1
            mc["accumulated_err"] += float(np.abs(1.0 - mc["accumulated_ratio"]))

            mapped = int(np.round(mc["cnt"] * ((28 - 1) / (mc["num_steps"] - 1)))) if mc["num_steps"] > 1 else mc["cnt"]

            if (mc["accumulated_err"] <= mc["magcache_thresh"]
                and mc["accumulated_steps"] <= mc["K"]
                and mapped != 11):
                skip_forward = True
            else:
                mc["accumulated_ratio"] = 1.0
                mc["accumulated_steps"] = 0
                mc["accumulated_err"] = 0.0

        if skip_forward:
            current["type"] = "MagCacheSkip"
        else:
            current["type"] = "full"
            current.setdefault("activated_steps", []).append(step)

        mc["cnt"] = (mc["cnt"] + 1) % max(1, T)
        return

        
    if cache_dic.get("mode") == "FasterCache":
        # =========================
        # FasterCache branch (CFG skip / dynamic reuse scheduler)
        # =========================
        fc = cache_dic.get("fastercache", {}).get("cfg", None)
        fr = cache_dic.get("fastercache", {}).get("reuse", None)
        step = int(current.get("step", 0))
        T = int(current.get("num_steps", 0) or 0)

        # -------------------------
        # (A) CFG-skip scheduler
        # -------------------------
        cfg_warmup = T // 3
        cfg_period = int(fc.get("refresh_period", 5)) if (fc and fc.get("enabled", False)) else 1
        cfg_period = max(1, cfg_period)
        if (fc is not None) and fc.get("enabled", False):
            if step < cfg_warmup:
                current["cfg_type"] = "cfg_full"         # 必跑 uncond
                current.setdefault("activated_steps", []).append(step)
            else:
                if ((step - cfg_warmup) % cfg_period) == 0:
                    current["cfg_type"] = "cfg_full"
                    current.setdefault("activated_steps", []).append(step)
                else:
                    current["cfg_type"] = "cfg_skip"     # 用 residual 近似 uncond
        else:
            current["cfg_type"] = "cfg_full"             # 未启用 FasterCache CFG 时，默认 full

        # -------------------------
        # (B) DFR(attn reuse) scheduler  [paper-faithful]
        # full attn every 2 timesteps from the beginning
        # Need first two full steps to build prev/prev2
        # -------------------------
        dfr_period = int(fr.get("refresh_period", 2)) if (fr and fr.get("enabled", False)) else 2
        if (fr is not None) and fr.get("enabled", False) and T > 0:
            # enforce first two full to have prev & prev2
            if step < 2:
                current["reuse_type"] = "attn_full"
            else:
                # alternate: even steps full, odd steps reuse
                current["reuse_type"] = "attn_full" if (step % dfr_period == 0) else "attn_reuse"
        else:
            current["reuse_type"] = "attn_full"

        # -------------------------
        # (C) DFR weight w(t): linear ramp from start of reuse -> end
        # -------------------------
        if (fr is not None) and fr.get("enabled", False) and T > 0:
            reuse_begin = 3  # step 3 is the first reuse if step0/1 are full and step2 is full
            end = max(reuse_begin + 1, T - 1)
            w = (step - reuse_begin) / float(end - reuse_begin)
            current["dfr_w"] = float(min(1.0, max(0.0, w)))
        else:
            current["dfr_w"] = 0.0

        current["type"] = "full"
        return
    # 🔥 修复：在 original 模式下，所有步骤都应该是 'full' 类型
    if (
        (cache_dic["fresh_ratio"] == 0.0)
        and (not cache_dic["taylor_cache"])
        and (cache_dic["fresh_threshold"] == 1)
    ):
        # Original mode: 每一步都进行完整计算，无缓存
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 ORIGINAL 模式 -> type='full'")
        return

    # 🔥 修复：正确实现first_enhance逻辑
    # first_enhance期间的所有步骤都应该是full模式
    in_first_enhance_period = current["step"] < cache_dic["first_enhance"]

    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA:Uniform
        first_step = current["step"] == 0
    else:
        # ToCa: First enhanced - 前first_enhance步都是full模式
        first_step = in_first_enhance_period

    force_fresh = cache_dic["force_fresh"]
    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        # current['activated_times'].append(current['t'])
        force_scheduler(cache_dic, current)
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: first_step={first_step}, 选择 -> type='full'")

    elif cache_dic["taylor_cache"]:
        cache_dic["cache_counter"] += 1
        if cache_dic.get("cluster_num", 0) > 0:
            current["type"] = "ClusCa"
        else:
            current["type"] = "taylor_cache"
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 TAYLOR_CACHE 模式 -> type='taylor_cache'")

    elif cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    # 'cache_noise' 'ToCa' 'FORA'
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
        # if current['step'] < 25:
        #    current['type'] = 'FORA'
        # else:
        #    current['type'] = 'aggressive'


######################################################################
# if (current['step'] in [3,2,1,0]):
#    current['type'] = 'full'
