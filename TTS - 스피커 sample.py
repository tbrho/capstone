from gtts import gTTS
from google.cloud import storage
import os
import uuid
import pygame

# 수정필요
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pi/your_key.json"
BUCKET_NAME = "a_sample"

# GCS 업로드 함수
def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"GCS에 업로드 완료: gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"

# TTS + 재생 + 업로드 함수
def handle_tts_and_play(text):
    safe_text = text.replace(" ", "_")
    filename = f"EX_{safe_text}_{uuid.uuid4().hex[:6]}.mp3"
    local_path = os.path.join("/home/pi", filename)

    # TTS 변환
    tts = gTTS(text=text, lang='ko')
    tts.save(local_path)
    print(f"TTS 파일 저장: {local_path}")

    # GCS 업로드
    upload_to_gcs(local_path, BUCKET_NAME, filename)

    # 스피커로 재생
    pygame.mixer.init()
    pygame.mixer.music.load(local_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    print("재생 완료")

if __name__ == "__main__":
    try:
        text = input("TTS로 변환할 텍스트를 입력하세요: ").strip()
        if text:
            handle_tts_and_play(text)
        else:
            print("입력된 텍스트가 없습니다.")
    except KeyboardInterrupt:
        print("\n프로그램 종료됨.")
