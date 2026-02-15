export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://bamboo-proxy.jd.com:80 https_proxy=http://bamboo-proxy.jd.com:80
python evaluation/evaluate.py \
  --ref_dir /export/home/liuyiming54/CFGCache/results/original/original_auto_20260214_172541/mn_flux-dev_i_5_o_2_s_50_hs_0.6 \
  --cmp_dir /export/home/liuyiming54/CFGCache/results/Taylor/taylor_auto_20260215_120955/mn_flux-dev_i_7_o_2_s_50_hs_0.5\
  --prompt_file resources/prompts/prompt.txt \
  --device cuda:5 \
  --prompt_align by_index
