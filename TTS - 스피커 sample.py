import os
from google.cloud import speech
from gtts import gTTS
import pygame
import time
from pydub import AudioSegment
import serial  # 아두이노와 통신을 위한 시리얼 모듈 추가

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LG\Desktop\2025\구글 클라우드\turnkey-channel-454904-b0-33a5c7635740.json"

def connect_jdy62(port="COM5", baudrate=9600):
    """JDY-62 블루투스 모듈과 연결"""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print("JDY-62 블루투스 모듈과 연결되었습니다.")
        return ser
    except Exception as e:
        print(f"JDY-62 연결 오류: {e}")
        return None

def send_audio_to_jdy62(serial_conn, file_path):
    """JDY-62 블루투스 모듈을 통해 오디오 데이터 전송"""
    if serial_conn:
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                serial_conn.write(data)
                print("JDY-62를 통해 블루투스 스피커로 전송 완료.")
        except Exception as e:
            print(f"JDY-62 데이터 전송 오류: {e}")

def text_to_speech(text, lang='ko', save_dir='C:/Users/LG/Desktop/2025/구글 클라우드', filename='output.mp3', jdy62_port=None):
    # 절대경로 설정
    mp3_filename = os.path.join(save_dir, filename)
    
    # 텍스트를 MP3 파일로 변환
    tts = gTTS(text=text, lang=lang)
    tts.save(mp3_filename)
    
    print(f"음성 파일이 {mp3_filename} 로 저장되었습니다.")
    
    # JDY-62 블루투스 모듈을 통해 전송
    if jdy62_port:
        serial_conn = connect_jdy62(jdy62_port)
        if serial_conn:
            send_audio_to_jdy62(serial_conn, mp3_filename)
            serial_conn.close()
    else:
        # pygame을 사용하여 음성 출력
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_filename)
        pygame.mixer.music.play()
        
        # 음성이 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    
    return mp3_filename

if __name__ == "__main__":
    text = input("앵무새에게 들려줄 문장을 입력하세요: ")
    jdy62_port = "COM5"  # JDY-62 블루투스 모듈이 연결된 포트 설정
    text_to_speech(text, jdy62_port=jdy62_port)
