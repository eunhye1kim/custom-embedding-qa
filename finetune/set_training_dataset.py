from datasets import load_dataset, Dataset, concatenate_datasets
import json
from dotenv import load_dotenv
import os

# .env 파일에서 환경변수 불러오기
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")

# 경로 설정
output_dir = "./finetune/data"
os.makedirs(output_dir, exist_ok=True)

from huggingface_hub import login
login(HF_TOKEN)  # .env에서 불러온 access token 사용

# 1. Hugging Face에서 기존 split 로드
base_dataset = load_dataset(HF_DATASET_NAME, split="train")
test_dataset = load_dataset(HF_DATASET_NAME, split="test")

# 2. slang_scenarios.jsonl 로딩
slang_data = []
with open(f"{output_dir}/slang_scenarios.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        # 🔧 source_id 강제 문자열 처리
        if isinstance(item["meta"]["source_id"], int):
            item["meta"]["source_id"] = str(item["meta"]["source_id"])
        slang_data.append(item)

slang_ds = Dataset.from_list(slang_data)

# 3. 병합 (train만 대상)
train_combined = concatenate_datasets([base_dataset, slang_ds])

# 4. JSONL 저장 함수
def save_jsonl(dataset: Dataset, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

# 5. 저장
save_jsonl(train_combined, f"{output_dir}/train.jsonl")
save_jsonl(test_dataset, f"{output_dir}/valid.jsonl")

print("✅ 저장 완료: ./finetune/data/train.jsonl, ./finetune/data/valid.jsonl")
