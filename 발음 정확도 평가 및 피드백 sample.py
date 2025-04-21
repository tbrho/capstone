import librosa
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI

# OpenAI API 키 설정
client = OpenAI(
    api_key="")


# MFCC 추출 함수
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)  # 파일 로드
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCC 추출
    return np.mean(mfcc, axis=1)  # 평균값을 구하여 특징 벡터로 반환

# 두 음성 파일 비교 함수
reference_file = 'C:/Users/LG/Desktop/2025/구글 클라우드/sample_reference.wav'
test_file = 'C:/Users/LG/Desktop/2025/구글 클라우드/sample.wav'
def compare_pronunciation(reference_file, test_file):
    mfcc1 = extract_mfcc(reference_file)  # 기준 파일의 MFCC
    mfcc2 = extract_mfcc(test_file)       # 앵무새 파일의 MFCC
    similarity = 1 - cosine(mfcc1, mfcc2)  # 코사인 유사도로 발음 비교
    return similarity

# 발음 유사도 계산
reference_file = 'C:/Users/LG/Desktop/2025/구글 클라우드/sample_reference.wav'
test_file = 'C:/Users/LG/Desktop/2025/구글 클라우드/sample.wav'
score = compare_pronunciation(reference_file, test_file)
print(f"발음 유사도: {score:.2f}")

# GPT API로 발음 피드백 받기
def get_feedback(word, score):
    prompt = f"""
    앵무새가 '{word}'을(를) 따라 말했을 때 발음 유사도가 {score*100:.1f}%였습니다.
    이 결과를 바탕으로 발음 피드백을 제공해 주세요.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 발음 교정 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# 피드백 출력
feedback = get_feedback("안녕 반가워", score)
print(f"피드백: {feedback}")
