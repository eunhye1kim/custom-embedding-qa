from datasets import Dataset, DatasetDict
from pathlib import Path
import json
from dotenv import load_dotenv
import os

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수에서 값 읽기
TRAIN_DIR = os.getenv("TRAIN_DIR")
TEST_DIR = os.getenv("TEST_DIR")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
HF_PRIVATE = os.getenv("HF_PRIVATE", "True").lower() == "true"

# 1. 고객 첫 발화 추출 함수
def extract_first_customer_query(content: str) -> str:
    for line in content.split("\n"):
        if line.startswith("고객:"):
            return line.replace("고객:", "").strip()
    return ""

# 2. 디렉토리 내 JSON 파일을 전처리하여 샘플 리스트로 반환
def load_and_convert_from_directory(directory: Path):
    all_samples = []
    for json_file in directory.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 실패: {json_file}")
                continue

        for item in data_list:
            consulting_content = item.get("consulting_content", "").strip()
            consulting_category = item.get("consulting_category", "").strip()
            if not consulting_content:
                continue

            query = extract_first_customer_query(consulting_content)
            if not query:
                continue

            sample = {
                "query": query,
                "positive_context": consulting_content,
                "meta": {
                    "consulting_category": consulting_category,
                    "source_id": item.get("source_id", "")
                }
            }
            all_samples.append(sample)
    return all_samples

# 3. 디렉토리 경로 지정
train_dir = Path(TRAIN_DIR)
test_dir = Path(TEST_DIR)

# 4. 전처리 후 데이터 로딩
train_data = load_and_convert_from_directory(train_dir)
test_data = load_and_convert_from_directory(test_dir)

# 5. Hugging Face Datasets 객체로 변환
train_ds = Dataset.from_list(train_data)
test_ds = Dataset.from_list(test_data)

# 6. DatasetDict 구성
dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds
})

# 7. Hugging Face Hub로 push
from huggingface_hub import login
login(HF_TOKEN)  # .env에서 불러온 access token 사용

dataset_dict.push_to_hub(HF_DATASET_NAME, private=HF_PRIVATE)
