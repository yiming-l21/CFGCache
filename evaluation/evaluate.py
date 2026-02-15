import os
import re
import math
import json
from pathlib import Path

os.environ["CLEANFID_CACHE_DIR"] = "/export/home/liuyiming54/inception_model"

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import lpips

# ---- CLIP ----
import open_clip
import torch.nn.functional as F


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def psnr_u8(ref_u8: np.ndarray, cmp_u8: np.ndarray) -> float:
    ref = ref_u8.astype(np.float32) / 255.0
    cmp = cmp_u8.astype(np.float32) / 255.0
    mse = np.mean((ref - cmp) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def to_lpips_tensor(u8: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(u8).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0) * 2.0 - 1.0
    return x.to(device)


def read_prompts(prompt_file: str) -> list[str]:
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    if not prompts:
        raise RuntimeError(f"Empty prompt_file: {prompt_file}")
    return prompts


def parse_index_from_name(name: str) -> int:
    """
    Try to parse an integer index from filename like img_0007.png -> 7.
    Uses the LAST integer group in the stem.
    """
    stem = Path(name).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Cannot parse index from filename: {name}")
    return int(nums[-1])


@torch.no_grad()
def clip_text_image_score(
    pil_img: Image.Image,
    prompt: str,
    clip_model,
    preprocess,
    tokenizer,
    device: str,
) -> float:
    img_in = preprocess(pil_img).unsqueeze(0).to(device)
    txt_in = tokenizer([prompt]).to(device)

    img_feat = clip_model.encode_image(img_in)
    txt_feat = clip_model.encode_text(txt_in)

    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)

    # cosine similarity
    sim = (img_feat * txt_feat).sum(dim=-1).item()
    return float(sim)


def main(
    ref_dir: str,
    cmp_dir: str,
    prompt_file: str,
    out_json: str = "metrics.json",
    out_csv: str = "metrics.csv",
    device: str = "cuda",
    resize_if_mismatch: bool = True,
    prompt_align: str = "by_index",   # by_index | by_order
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
):
    ref_dir = Path(ref_dir)
    cmp_dir = Path(cmp_dir)
    assert ref_dir.is_dir(), f"ref_dir not found: {ref_dir}"
    assert cmp_dir.is_dir(), f"cmp_dir not found: {cmp_dir}"

    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Collect matched filenames (by name)
    ref_files = {p.name: p for p in sorted(ref_dir.glob("*.jpg"))}
    cmp_files = {p.name: p for p in sorted(cmp_dir.glob("*.jpg"))}
    names = sorted(set(ref_files.keys()) & set(cmp_files.keys()))
    if not names:
        raise RuntimeError("No matched *.jpg filenames between ref_dir and cmp_dir")

    prompts = read_prompts(prompt_file)

    # LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    # CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    rows = []
    psnrs, lpipss = [], []
    clip_ref_list, clip_cmp_list = [], []

    for order_i, name in enumerate(tqdm(names, desc="Pairwise PSNR/LPIPS + CLIPScore")):
        ref_u8 = load_rgb(ref_files[name])
        cmp_u8 = load_rgb(cmp_files[name])

        if ref_u8.shape != cmp_u8.shape:
            if not resize_if_mismatch:
                raise RuntimeError(f"Shape mismatch for {name}: {ref_u8.shape} vs {cmp_u8.shape}")
            cmp_img = Image.fromarray(cmp_u8).resize((ref_u8.shape[1], ref_u8.shape[0]), Image.BICUBIC)
            cmp_u8 = np.array(cmp_img)

        # ---- prompt alignment ----
        if prompt_align == "by_index":
            idx = parse_index_from_name(name)
        elif prompt_align == "by_order":
            idx = order_i
        else:
            raise ValueError(f"Unknown prompt_align={prompt_align}")

        if idx < 0 or idx >= len(prompts):
            raise IndexError(f"Prompt index out of range for {name}: idx={idx}, num_prompts={len(prompts)}")

        prompt = prompts[idx]

        # ---- PSNR ----
        p = psnr_u8(ref_u8, cmp_u8)

        # ---- LPIPS ----
        with torch.no_grad():
            ref_t = to_lpips_tensor(ref_u8, device)
            cmp_t = to_lpips_tensor(cmp_u8, device)
            l = float(lpips_model(ref_t, cmp_t).item())

        # ---- CLIPScore(text-image) for BOTH dirs ----
        ref_pil = Image.fromarray(ref_u8)
        cmp_pil = Image.fromarray(cmp_u8)
        cs_ref = clip_text_image_score(ref_pil, prompt, clip_model, preprocess, tokenizer, device)
        cs_cmp = clip_text_image_score(cmp_pil, prompt, clip_model, preprocess, tokenizer, device)

        rows.append(
            {
                "name": name,
                "prompt_idx": idx,
                "prompt": prompt,
                "psnr": p,
                "lpips": l,
                "clip_ref": cs_ref,
                "clip_cmp": cs_cmp,
            }
        )

        psnrs.append(p if np.isfinite(p) else 1e9)
        lpipss.append(l)
        clip_ref_list.append(cs_ref)
        clip_cmp_list.append(cs_cmp)

    # Summary stats
    psnr_mean = float(np.mean(psnrs))
    psnr_std = float(np.std(psnrs))
    lpips_mean = float(np.mean(lpipss))
    lpips_std = float(np.std(lpipss))

    clip_ref_mean = float(np.mean(clip_ref_list))
    clip_ref_std = float(np.std(clip_ref_list))
    clip_cmp_mean = float(np.mean(clip_cmp_list))
    clip_cmp_std = float(np.std(clip_cmp_list))

    # FID (set-level)
    fid_value = None
    try:
        from cleanfid import fid
        fid_value = float(
            fid.compute_fid(
                str(ref_dir),
                str(cmp_dir),
                mode="clean",
                device=device,
                use_dataparallel=False,
            )
        )
    except Exception as e:
        fid_value = None
        print(f"[WARN] FID not computed (cleanfid failed): {e}")

    summary = {
        "ref_dir": str(ref_dir),
        "cmp_dir": str(cmp_dir),
        "prompt_file": prompt_file,
        "prompt_align": prompt_align,
        "num_pairs": len(names),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "fid": fid_value,
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "clip_ref_mean": clip_ref_mean,
        "clip_ref_std": clip_ref_std,
        "clip_cmp_mean": clip_cmp_mean,
        "clip_cmp_std": clip_cmp_std,
        "device": device,
    }

    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": rows}, f, ensure_ascii=False, indent=2)

    # Save CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,prompt_idx,psnr,lpips,clip_ref,clip_cmp\n")
        for r in rows:
            f.write(f"{r['name']},{r['prompt_idx']},{r['psnr']},{r['lpips']},{r['clip_ref']},{r['clip_cmp']}\n")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {out_json}, {out_csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", type=str, required=True)
    ap.add_argument("--cmp_dir", type=str, required=True)
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="metrics.json")
    ap.add_argument("--out_csv", type=str, default="metrics.csv")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_resize_if_mismatch", action="store_true")

    ap.add_argument("--prompt_align", type=str, default="by_index", choices=["by_index", "by_order"])
    ap.add_argument("--clip_model", type=str, default="ViT-B-32")
    ap.add_argument("--clip_pretrained", type=str, default="openai")

    args = ap.parse_args()

    main(
        ref_dir=args.ref_dir,
        cmp_dir=args.cmp_dir,
        prompt_file=args.prompt_file,
        out_json=args.out_json,
        out_csv=args.out_csv,
        device=args.device,
        resize_if_mismatch=not args.no_resize_if_mismatch,
        prompt_align=args.prompt_align,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
