import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from google.cloud import storage

# Google Cloud 인증 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"

# GCS에서 파일 다운로드 함수
def download_from_gcs(bucket_name, blob_name, local_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"GCS에서 다운로드 완료: {local_path}")
    return local_path

# MFCC 추출 함수
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# 발음 비교 함수
def compare_pronunciation(reference_file, test_file):
    mfcc1 = extract_mfcc(reference_file)
    mfcc2 = extract_mfcc(test_file)
    similarity = 1 - cosine(mfcc1, mfcc2)
    return similarity

# === 실행 부분 ===
if __name__ == "__main__":
    # GCS에서 기준 파일 다운로드
    BUCKET_NAME = "a_sample"
    GCS_BLOB_NAME = "EX_안녕.mp3"  # GCS에 저장된 참조 음성 파일 이름
    local_reference_path = "EX_안녕.wav"
    download_from_gcs(BUCKET_NAME, GCS_BLOB_NAME, local_reference_path)

    # 테스트할 앵무새 음성 파일 경로
    test_file = "C:/Users/LG/Desktop/2025/구글 클라우드/앵무새 음성파일/안녕/안녕2.wav"

    # 유사도 측정
    score = compare_pronunciation(local_reference_path, test_file)
    print(f"발음 유사도: {score*100:.2f}%")
