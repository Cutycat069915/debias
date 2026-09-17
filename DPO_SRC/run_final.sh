#!/bin/bash
# Train and evaluate the final configuration (report_0902 §11), one seed at a time.
#
#   bash run_final.sh              # seeds 15 22 23 31 432
#   bash run_final.sh 22           # just one
#
# Run make_data.py first. Each seed: train ~16 min, eval ~10 min on one GPU.
# A seed whose adapter already exists skips training.
cd "$(dirname "$0")"
MODEL=models/Llama-3.2-11B-Vision-Instruct
SEEDS=${*:-15 22 23 31 432}
mkdir -p save results

for s in $SEEDS; do
  A=save/dpo_final_$s
  if [ ! -f "$A/adapter_config.json" ]; then
    echo "[$(date +%H:%M:%S)] train seed $s"
    python train_dpo.py --model_name_or_path "$MODEL" \
      --dataset_path "data/${s}_chat916" --output_dir "$A" \
      || { echo "!! FAIL train $s"; continue; }
    # Only the LoRA adapter is needed.
    rm -f "$A"/model.safetensors "$A"/config.json "$A"/generation_config.json
    rm -rf "$A"/checkpoint-*
  fi

  echo "[$(date +%H:%M:%S)] eval seed $s"
  python eval_bbq_official.py "data/${s}_pure916" "$A" DPO-final \
    "results/official_final_$s.json" Gender_identity --chat || echo "!! FAIL eval $s"
  # --chat on both: the adapter is trained on chat-template prompts, and scoring it
  # on raw concatenation costs ~9.5 MCQ points (report_0902 §11.3).
  python eval_mcq3.py "data/${s}_pure916" "$A" final \
    "results/mcq3_final_$s.json" --chat || echo "!! FAIL mcq3 $s"
done
