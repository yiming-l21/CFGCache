export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://bamboo-proxy.jd.com:80 https_proxy=http://bamboo-proxy.jd.com:80
export CUDA_VISIBLE_DEVICES=3
python evaluation/evaluate.py \
  --ref_dir /export/home/liuyiming54/result-cfg/flux/origin \
  --cmp_dir /export/home/liuyiming54/result-cfg/flux/teacache_1.0\
  --prompt_file resources/prompts/prompt.txt \
  --prompt_align by_index
