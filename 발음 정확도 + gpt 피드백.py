import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from google.cloud import storage
import openai  # OpenAI SDK

# ——— 환경 설정 ———
# Google Cloud 인증 경로
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"
)
BUCKET_NAME = "a_sample"

# OpenAI API Key 환경 변수에서 불러오기
openai.api_key = os.getenv("OPENAI_API_KEY")

# ——— GCS 다운로드 함수 ———
def download_from_gcs(bucket_name: str, blob_name: str, local_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

# ——— MFCC 추출 함수 ———
def extract_mfcc(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# ——— 발음 유사도 계산 ———
def compare_pronunciation(ref_path: str, test_path: str) -> float:
    mfcc1 = extract_mfcc(ref_path)
    mfcc2 = extract_mfcc(test_path)
    similarity = 1 - cosine(mfcc1, mfcc2)
    return similarity

# ——— GPT 피드백 생성 함수 ———
def generate_feedback(word: str, accuracy: float) -> str:
    prompt = (
        f"사용자가 '{word}'를 발음했을 때 유사도 {accuracy:.2f}%가 나왔습니다. "
        "이 발음을 어떻게 개선하면 좋을지, 구체적이고 친절하게 조언해주세요."
    )

    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()


# ——— 메인 실행 로직 ———
if __name__ == "__main__":
    # 테스트용 변수 (원하는 값으로 수정)
    reference_gcs = "EX_안녕하세요.mp3"
    test_local_path = r"C:\Users\LG\Desktop\앵무새 음성파일\안녕하세요\안녕하세요.wav"   # 일단 로컬위치에서 가져옴, 추후 스피커+마이크+발음정확도+피드백 합치기
    local_ref_path = "temp_ref.mp3"

    # 1) GCS에서 기준 음성 다운로드
    try:
        download_from_gcs(BUCKET_NAME, reference_gcs, local_ref_path)
        print(f"✅ 기준 음성 다운로드: {local_ref_path}")
    except Exception as e:
        print(f"❌ GCS 다운로드 실패: {e}")
        exit(1)

    # 2) 발음 유사도 계산
    try:
        score = compare_pronunciation(local_ref_path, test_local_path)
        accuracy = round(score * 100, 2)
        print(f"🎯 발음 유사도: {accuracy}%")
    except Exception as e:
        print(f"❌ 발음 비교 실패: {e}")
        exit(1)

    # 3) GPT 피드백 생성
    try:
        word = reference_gcs.replace("EX_", "").replace(".mp3", "")
        feedback = generate_feedback(word, accuracy)
        print("💬 피드백:\n" + feedback)
    except Exception as e:
        print(f"❌ GPT 피드백 생성 실패: {e}")
        exit(1)
