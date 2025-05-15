python evaluate.py --dataset_name LitSearch \
  --gt_file /home/zhangshuaiyu/PerMed/data/LitSearch/query_to_texts.jsonl \
  --rerank_pred_file /home/zhangshuaiyu/PerMed/baselines/results/LitSearch/rpmn_results.jsonl \
  --unrerank_pred_file /home/zhangshuaiyu/PerMed/results/LitSearch/retrieved.jsonl


python evaluate.py --dataset_name MedCorpus \
  --gt_file /home/zhangshuaiyu/PerMed/data/MedCorpus/query_to_texts.jsonl \
  --rerank_pred_file /home/zhangshuaiyu/PerMed/baselines/results/MedCorpus/rpmn_results.jsonl \
  --unrerank_pred_file /home/zhangshuaiyu/PerMed/results/MedCorpus/retrieved.jsonl

