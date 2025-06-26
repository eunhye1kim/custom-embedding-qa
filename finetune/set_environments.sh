#!/bin/bash

cd ..

apt update && apt install git vim nano -y

python -m venv env
source env/bin/activate

pip install git+https://github.com/huggingface/autotrain-advanced.git
pip install datasets transformers accelerate peft bitsandbytes wandb python-dotenv

# .env 파일 로드
set -a
source .env
set +a

wandb login $WANDB_API_KEY

autotrain sentence-transformers \
  --train \
  --project-name $WANDB_PROJECT \
  --model google-bert/bert-base-multilingual-cased \
  --data-path ./data \
  --sentence1-column query \
  --sentence2-column positive_context \
  --trainer pair \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 32 \
  --eval-strategy epoch \
  --early-stopping-patience 2 \
  --early-stopping-threshold 0.0001 \
  --push-to-hub \
  --username $HF_USERNAME \
  --token $HF_TOKEN \
  --log wandb
