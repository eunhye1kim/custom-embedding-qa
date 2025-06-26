import os
import json
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수에서 값 읽기
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

if not INPUT_DIR or not OUTPUT_DIR:
    raise ValueError("INPUT_DIR와 OUTPUT_DIR 환경변수를 .env에 지정해야 합니다.")

input_dir = Path(INPUT_DIR)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# JSON 파일에서 consulting_content와 instruction 추출 및 txt로 저장
def convert_json_to_txt(input_dir: Path, output_dir: Path):
    for json_file in input_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 실패: {json_file}")
                continue

        for item in data_list:
            source_id = item.get("source_id", "unknown")
            # consulting_content 저장
            content = item.get("consulting_content", "").strip()
            if content:
                content_path = output_dir / f"{source_id}_content.txt"
                with open(content_path, "w", encoding="utf-8") as out_f:
                    out_f.write(content)

if __name__ == "__main__":
    convert_json_to_txt(input_dir, output_dir) 