import os
import re
from google.cloud import speech
from google.cloud import storage
from io import BytesIO

# 1. 인증 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"

# 2. 클라이언트 생성
speech_client = speech.SpeechClient()
storage_client = storage.Client()

# 3. 음성 파일 로드
audio_path = 'C:/Users/LG/Desktop/2025/구글 클라우드/앵무새 음성파일/안녕하세요2.wav'
with open(audio_path, 'rb') as audio_file:
    audio_content = audio_file.read()

audio = speech.RecognitionAudio(content=audio_content)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    language_code="ko-KR"
)

# 4. STT 실행
response = speech_client.recognize(config=config, audio=audio)

# 5. 결과 텍스트 합치기
transcript_text = ""
for result in response.results:
    transcript_text += result.alternatives[0].transcript + "\n"

print("Transcript:\n", transcript_text)

# 6. 텍스트를 GCS-safe 파일명으로 변환
def sanitize_filename(text):
    text = text.strip()
    text = re.sub(r'[\\/*?:"<>|#\n\r]', "_", text)  # 금지 문자 제거
    text = text[:100]  # 길이 제한
    return text if text else "empty_result"

safe_filename = sanitize_filename(transcript_text)

# 7. Google Cloud Storage 업로드
bucket_name = "a_sample"
bucket = storage_client.bucket(bucket_name)

# ① 텍스트 파일 업로드
text_blob = bucket.blob(f"{safe_filename}.txt")
text_stream = BytesIO(transcript_text.encode("utf-8"))
text_blob.upload_from_file(text_stream, content_type="text/plain")

# ② 음성 파일 업로드
audio_blob = bucket.blob(f"{safe_filename}.wav")
audio_stream = BytesIO(audio_content)
audio_blob.upload_from_file(audio_stream, content_type="audio/wav")

print(f"✅ GCS에 다음 파일들이 저장되었습니다:")
print(f" - 텍스트: gs://{bucket_name}/{safe_filename}.txt")
print(f" - 음성: gs://{bucket_name}/{safe_filename}.wav")
