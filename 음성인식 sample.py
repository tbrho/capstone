import os
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"


# Google Cloud Speech-to-Text API 사용을 위한 클라이언트 객체 생성
client = speech.SpeechClient()

# 음성 파일 읽기

with open('C:/Users/LG/Desktop/2025/구글 클라우드/sample.wav', 'rb') as audio_file:
    content = audio_file.read()

# 음성 파일에 대한 오디오 객체 생성
audio = speech.RecognitionAudio(content=content)

# 음성 인식 설정
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=24000,
    language_code="ko-KR"
)

# 음성을 텍스트로 변환하는 요청
response = client.recognize(config=config, audio=audio)

# 인식된 텍스트 출력
for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))