# custom-embedding-qa (한국어)

> 커스텀 임베딩 기반 의미 유사 질의응답 QA 시스템
> OpenAI, HuggingFace 임베딩 모델 비교 및 파인튜닝 실험 포함

---

## 📌 개요

이 프로젝트는 RAG(Retrieval-Augmented Generation) 시스템에서 발생하는 표현 다양성 및 문서 중복 문제를 해결하기 위해, 커스텀 임베딩을 활용한 의미 기반 검색 QA 시스템을 개발하는 것입니다.

Streamlit을 기반으로 UI를 제공하며, LangChain, Qdrant, HuggingFace 임베딩 등을 통합하여 검색 품질을 실험합니다.

---

## 🤩 주요 기능

* OpenAI 임베딩 기반 QA 시스템
* HuggingFace 모델(BGE, MiniCoIL 등) 기반 질의응답
* Streamlit 챗봇 인터페이스
* 커스텀 임베딩 모델 파인튜닝 및 서빙 환경 제공
* 검색 품질 비교 실험 (NDCG, 정성 평가 등)

---

## 💪 기술 스택

| 분류     | 도구                              |
| ------ | ------------------------------- |
| 언어     | Python                          |
| 프레임워크  | LangChain, Streamlit            |
| 임베딩 모델 | OpenAI, MiniCoIL, SBERT         |
| 벡터 DB  | Qdrant                          |
| 개발 환경  | Colab, VSCode, Jupyter Notebook |

---

## 📁 프로젝트 구조

```
custom-embedding-qa/
├── app/             # Streamlit UI
├── backend/         # 임베딩 및 검색 로직
├── data/            # 문서 데이터 및 벡터 저장소
├── finetune/        # 모델 파인튜닝 및 서빙 코드
├── config/          # 설정 파일
├── requirements.txt
├── README.en.md     # 영어 버전
└── README.ko.md     # 한국어 버전
```

---

## 🚀 실행 방법

1. 패키지 설치

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정
   `.env.example` 파일을 `.env`로 복사한 후, API 키 등을 입력합니다.

3. Streamlit 실행 시 PYTHONPATH 환경변수 지정 (모듈 import 오류 방지)

**Windows (CMD):**
```bash
set PYTHONPATH=.
streamlit run app/main.py
```
**Windows (PowerShell):**
```bash
$env:PYTHONPATH="."
streamlit run app/main.py
```
**macOS/Linux:**
```bash
PYTHONPATH=. streamlit run app/main.py
```

---

## 👩‍💻 작성자

김은혜
Machine Learning Engineer / AI 시스템 빌더
목표: 커스텀 임베딩 기반 의미 검색 시스템을 실제 실험하고 포트폴리오로 정리

---

## 📘 다른 언어

📘 [영문 README 보기](README.en.md)
