import os
import uuid
import wave
import pyaudio
from gtts import gTTS
import pygame
import re
from flask import Flask, jsonify
from google.cloud import storage
import openai
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)

# OpenAI API 키
openai.api_key = ''

# GCP 인증 경로
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
BUCKET_NAME = "a_sample"

# 📁 오디오 데이터 폴더 경로
audio_dir = ''

# 녹음 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10

# ————— 파일명 안전 처리 함수 —————
def sanitize(text: str) -> str:
    """파일명으로 안전하게 사용할 수 있도록 변환 (<> 제외)"""
    text = text.strip()
    # 허용되지 않는 문자: \ / * ? : " < > | 
    text = re.sub(r'[\\/*?:"<>|]', "_", text)
    return text or "untitled"

# ————— GCS 업로드 함수 —————
def upload_to_gcs(local_path: str, blob_name: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"✅ GCS 업로드: gs://{BUCKET_NAME}/{blob_name}")

# ————— TTS 생성·재생·업로드 —————
def tts_and_play(text: str, safe: str):
    # TTS 파일명
    tts_filename = f"EX_{safe}.mp3"
    local_tts = os.path.join(os.path.expanduser("~"), tts_filename)
    # 1) TTS 변환
    tts = gTTS(text=text, lang='ko')
    tts.save(local_tts)

    # 2) GCS 업로드
    upload_to_gcs(local_tts, tts_filename)

    # 텍스트 파일 저장
    txt_filename = f"{safe}.txt"
    local_txt = os.path.join(os.path.expanduser("~"), txt_filename)
    with open(local_txt, "w", encoding="utf-8") as f:
        f.write(text)
    upload_to_gcs(local_txt, txt_filename)

    # 3) 스피커 재생
    pygame.mixer.init()
    pygame.mixer.music.load(local_tts)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    time.sleep(1)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    print("🔈 재생 완료")

# 반복학습-TTS 
def tts_repeat(text: str, safe: str):

    tts_filename = f"EX_{safe}.mp3"
    local_tts = os.path.join(os.path.expanduser("~"), tts_filename)

    tts = gTTS(text=text, lang='ko')
    tts.save(local_tts)

    pygame.mixer.init()
    pygame.mixer.music.load(local_tts)
    for i in range(10):
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # CPU 낭비 방지
        time.sleep(1)  # 1초 쉬고 다음 재생
    pygame.mixer.quit()
    print("🔈 재생 완료")



# ————— 녹음·저장·업로드 —————
def record_and_upload(safe: str):
    rec_filename = f"{safe}.wav"
    local_rec = os.path.join(os.path.expanduser("~"), rec_filename)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print(f"🎙️ {RECORD_SECONDS}초간 녹음 시작...")
    frames = [stream.read(CHUNK) for _ in range(int(RATE/CHUNK*RECORD_SECONDS))]
    print("⏹ 녹음 완료")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # WAV로 저장
    with wave.open(local_rec, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"💾 녹음 파일 저장: {local_rec}")

    # GCS 업로드
    upload_to_gcs(local_rec, rec_filename)




# 🎵 Mel-spectrogram 추출 함수
def extract_features(audio_path, target_length=130):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_resized = librosa.util.fix_length(mel_spec, size=target_length, axis=-1)
    mel_spec_db = librosa.power_to_db(mel_spec_resized, ref=np.max)
    return mel_spec_db

# 📚 데이터 로딩
def load_data(audio_dir):
    X, y = [], []
    for label in os.listdir(audio_dir):
        label_dir = os.path.join(audio_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith('.wav'):
                    features = extract_features(os.path.join(label_dir, filename))
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# 🧠 CNN 모델 학습
def train_and_save_model():
    X, y = load_data(audio_dir)
    print(f"✅ Loaded {len(X)} samples")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = X[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(np.unique(y_encoded)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save('parrot_speech_model.h5')
    print("✅ Model trained and saved as 'parrot_speech_model.h5'")
    return le

# 🔍 오디오 분류
def classify_new_audio(model, audio_path, label_encoder):
    features = extract_features(audio_path)
    features = np.expand_dims(features[..., np.newaxis], axis=0)
    prediction = model.predict(features)
    predicted_idx = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_idx)[0]
    return predicted_label

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


# 💬 GPT 피드백 생성
def get_feedback_from_gpt(word, accuracy):
    percent = int(accuracy * 100) 
    prompt = (
        f"앵무새가 '{word}' 라는 단어를 발음했습니다. "
        f"AI 모델이 판단한 발음 정확도는 {percent}%입니다. "
        f"이 정보를 바탕으로 앵무새에게 줄 수 있는 간단하고 친근한 피드백 문장을 작성해주세요. "
        f"예를 들어 '발음이 아주 좋아요!', '조금만 더 연습해볼까요?' 같은 형식으로요."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.7,
        n=1,
    )
    
    # 피드백 처리
    try:
        # 'choices'와 'message' 존재 여부 확인
        feedback = response['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError):
        feedback = "피드백을 받을 수 없습니다. 다시 시도해주세요."
    
    return feedback


# 레벨 테스트 
def load_all_words_from_gcs(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    learned_words = []
    for blob in storage_client.list_blobs(bucket):
        if blob.name.endswith(".txt"):
            text = blob.download_as_text()
            learned_words += [w.strip() for w in text.splitlines() if w.strip()]
    return learned_words

def evaluate_level_with_gpt(words):
    prompt = f"""
아래는 앵무새가 학습한 단어 목록이야:
{words}

중복된 단어는 감안해서 이 앵무새의 언어 수준을 1에서 10까지의 레벨로 평가해줘.
숫자만 하나로 답해줘.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"당신은 앵무새 언어학습 평가 전문가입니다."},
            {"role":"user","content":prompt}
        ]
    )
    return int(resp.choices[0].message["content"].strip())

def recommend_words_by_level(level):
    prompt = f"""
앵무새의 언어 레벨이 {level}이야.
이 수준에 맞는 한국어 단어 또는 짧은 문장 5개를 추천해줘.
리스트 형식으로 보여줘.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"당신은 앵무새 언어학습 단어 추천 전문가입니다."},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message["content"].strip()






@app.route("/api/parrot_status", methods=["GET"])
def parrot_status():
    bucket_name = BUCKET_NAME
    words = load_all_words_from_gcs(bucket_name)
    level = evaluate_level_with_gpt(words)
    recommended = recommend_words_by_level(level)
    return jsonify({
        "level": level,
        "recommended_words": recommended
    })


@app.route("/api/run_process", methods=['POST'])
def process_word():
    try:
        input_text = request.json.get('word', '').strip()
        if not input_text:
            return jsonify({"error": "입력된 단어가 없습니다."}), 400

        safe_text = sanitize(input_text)

        # 1. TTS 생성 + 재생 + GCS 업로드
        tts_and_play(input_text, safe_text)

        # 2. 녹음 + 저장 + GCS 업로드
        record_and_upload(safe_text)

        # 3. 모델 로드
        model_path = 'parrot_speech_model.h5'
        if not os.path.exists(model_path):
            label_encoder = train_and_save_model()
        else:
            model = tf.keras.models.load_model(model_path)
            label_encoder = LabelEncoder()
            label_encoder.fit(os.listdir(audio_dir))

        # 4. 분류
        new_audio_file = os.path.join(os.path.expanduser("~"), f"{safe_text}.wav")
        predicted_word = classify_new_audio(model, new_audio_file, label_encoder)

        # 발음 정확도
        reference_audio = os.path.join(os.path.expanduser("~"), f"EX_{safe_text}.mp3")
        accuracy = compare_pronunciation(reference_audio, new_audio_file)
    
        # 5. GPT 피드백 생성
        feedback = get_feedback_from_gpt(predicted_word, accuracy)

        # 6. 결과 반환
        return jsonify({
            "input_word": input_text,
            "predicted_word": predicted_word,
            "accuracy": round(accuracy, 3),
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/repeat", methods=['POST'])
def repeat_word():
    try:
        input_text = request.json.get('word', '').strip()
        if not input_text:
            return jsonify({"error": "입력된 단어가 없습니다."}), 400
        
        safe_text = sanitize(input_text)

        tts_repeat(input_text, safe_text)

        return jsonify({"repeat_result" : "반복 학습 완료"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
