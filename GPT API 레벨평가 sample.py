import os
import json
from google.cloud import storage
from openai import OpenAI

# ✅ OpenAI 클라이언트 생성 (api_key 직접 전달)
client = OpenAI(
    api_key=""  # ← 본인의 OpenAI API 키로 바꿔줘
)

# ✅ 구글 인증 정보
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"

# ✅ GCS에서 학습한 단어 파일 전부 읽기
def load_all_words_from_gcs(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    learned_words = []
    blobs = list(storage_client.list_blobs(bucket))

    for blob in blobs:
        if blob.name.endswith(".txt"):
            text = blob.download_as_text()
            words = [w.strip() for w in text.strip().splitlines() if w.strip()]
            learned_words.extend(words)

    print("📚 앵무새가 학습한 단어 목록:")
    for word in learned_words:
        print(f"- {word}")

    return learned_words

# ✅ GPT로 레벨 평가
def evaluate_level_with_gpt(words):
    prompt = f"""
    아래는 앵무새가 학습한 단어 목록이야:
    {words}

    이 앵무새의 언어 수준을 1에서 10까지의 레벨로 평가해줘.
    숫자만 하나로 답해줘.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 앵무새 언어학습 평가 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )

    return int(response.choices[0].message.content.strip())

# ✅ GPT로 추천 단어 받기
def recommend_words_by_level(level):
    prompt = f"""
    앵무새의 언어 레벨이 {level}이야.
    이 수준에 맞는 단어 또는 짧은 문장 5개를 추천해줘.
    리스트 형식으로 보여줘.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 앵무새 언어학습 단어 추천 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# ✅ 전체 실행 함수
def run_parrot_leveling(bucket_name):
    words = load_all_words_from_gcs(bucket_name)
    level = evaluate_level_with_gpt(words)
    recommendations = recommend_words_by_level(level)

    print(f"✅ 현재 앵무새의 언어 레벨: {level}")
    print(f"📝 추천 단어/문장:\n{recommendations}")

# ✅ 예시 실행
if __name__ == "__main__":
    bucket = "a_sample"  # 너의 버킷 이름
    run_parrot_leveling(bucket)
