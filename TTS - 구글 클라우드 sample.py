import os
from google.cloud import storage
from google.cloud import speech
from gtts import gTTS
import pygame
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"

BUCKET_NAME = "a_sample"

def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """로컬 파일을 GCS에 업로드"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"파일이 GCS에 업로드되었습니다: gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"

def text_to_speech(text, lang='ko', filename='output.mp3'):
    # 절대경로 설정
    local_path = os.path.join(os.getcwd(), filename)
    
    # 텍스트를 MP3 파일로 변환
    tts = gTTS(text=text, lang=lang)
    tts.save(local_path)
    
    print(f"음성 파일이 {local_path} 로 저장되었습니다.")
    
    # Google Cloud Storage에 업로드
    gcs_path = upload_to_gcs(local_path, BUCKET_NAME, filename)
    
    return gcs_path

if __name__ == "__main__":
    text = input("앵무새에게 들려줄 문장을 입력하세요: ")
    text_to_speech(text)
