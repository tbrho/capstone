import os
from google.cloud import speech
from google.cloud import storage

# 1. 인증 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"


# 2. 클라이언트 생성
speech_client = speech.SpeechClient()
storage_client = storage.Client()

# 3. 음성 파일 로드
with open('C:/Users/LG/Desktop/2025/구글 클라우드/sample.wav', 'rb') as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=24000,
    language_code="ko-KR"
)

# 4. STT 실행
response = speech_client.recognize(config=config, audio=audio)

# 5. 결과 텍스트 합치기
transcript_text = ""
for result in response.results:
    transcript_text += result.alternatives[0].transcript + "\n"

print("Transcript:\n", transcript_text)

# 6. 텍스트 파일로 저장 (선택적으로 로컬에도 저장 가능)
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript_text)

# 7. Google Cloud Storage에 업로드
bucket_name = "a_sample"  # ← 너의 GCS 버킷 이름으로 바꿔줘!
destination_blob_name = "sample_transcript.txt"

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)

blob.upload_from_filename("transcript.txt")

print(f"✅ 변환된 텍스트가 GCS에 저장되었습니다: gs://{bucket_name}/{destination_blob_name}")
