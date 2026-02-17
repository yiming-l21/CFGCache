export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://bamboo-proxy.jd.com:80 https_proxy=http://bamboo-proxy.jd.com:80
export CUDA_VISIBLE_DEVICES=3
python evaluation/evaluate.py \
  --ref_dir /export/home/liuyiming54/CFGCache/results/original/original_auto_20260216_132407/mn_flux-dev_i_7_o_2_s_50_hs_0.5 \
  --cmp_dir /export/home/liuyiming54/CFGCache/results/ClusCa/clusca_auto_20260216_143833/mn_flux-dev_i_7_o_2_s_50_hs_0.5\
  --prompt_file resources/prompts/prompt.txt \
  --prompt_align by_index
