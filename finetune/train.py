import os
import json
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 💡 환경 변수
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
HF_USERNAME = "lunara-kim"
HF_MODEL_REPO = "custom-embedding-slang-model"
OUTPUT_PATH = f"./output/{HF_MODEL_REPO}"

# ✅ W&B 설정
wandb.init(
    project="custom-embedding-qa",
    name="slang-finetune",
    job_type="training"
)

login(HF_TOKEN)  # .env에서 불러온 access token 사용

# ✅ 1. 데이터 로드 및 확장
base_ds = load_dataset(HF_DATASET_NAME, split="train")

with open("./slang_scenarios.jsonl", "r", encoding="utf-8") as f:
    slang_data = [json.loads(line) for line in f]
    for x in slang_data:
        x["meta"]["source_id"] = str(x["meta"]["source_id"])  # type 일치

slang_ds = Dataset.from_list(slang_data * 10)
combined_ds = concatenate_datasets([base_ds, slang_ds])

def to_input_examples(dataset):
    return [
        InputExample(texts=[row["query"], row["positive_context"]])
        for row in dataset
    ]

all_examples = to_input_examples(combined_ds)
slang_examples = to_input_examples(slang_ds)

# ✅ 2. 모델 구성
model_name = "google-bert/bert-base-multilingual-cased"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean'
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# ✅ 3. 학습 설정
train_loader_all = DataLoader(all_examples, shuffle=True, batch_size=32)
train_loader_slang = DataLoader(slang_examples, shuffle=True, batch_size=16)

loss_all = losses.MultipleNegativesRankingLoss(model)
loss_slang = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[
        (train_loader_all, loss_all),
        (train_loader_slang, loss_slang)
    ],
    epochs=5,
    warmup_steps=100,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

# ✅ 4. Hugging Face 업로드
model.push_to_hub(
    HF_USERNAME + "/" + HF_MODEL_REPO,
    token=HF_TOKEN
)

wandb.finish()

print(f"✅ 모델이 업로드되었습니다: https://huggingface.co/{HF_USERNAME}/{HF_MODEL_REPO}")
