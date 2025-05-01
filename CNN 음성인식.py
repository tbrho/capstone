import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 오디오 파일 경로
audio_dir = 'C:/Users/LG/Desktop/2025/구글 클라우드/앵무새 음성파일'

# 오디오 파일을 mel-spectrogram으로 변환하는 함수
def extract_features(audio_path, target_length=130):
    # 오디오 파일 읽기
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Mel-spectrogram 추출
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # 스펙트로그램의 길이를 target_length로 맞추기 (패딩 또는 자르기)
    mel_spec_resized = librosa.util.fix_length(mel_spec, size=target_length, axis=-1)
    
    # 로그 스케일로 변환
    mel_spec_db = librosa.power_to_db(mel_spec_resized, ref=np.max)
    return mel_spec_db

# 오디오 파일과 레이블을 로드하는 함수
def load_data(audio_dir):
    X = []
    y = []
    
    # 각 폴더를 레이블로 처리
    for label in os.listdir(audio_dir):
        label_dir = os.path.join(audio_dir, label)
        if os.path.isdir(label_dir):  # 폴더만 처리
            for filename in os.listdir(label_dir):
                if filename.endswith('.wav'):
                    # 오디오 파일 경로
                    audio_path = os.path.join(label_dir, filename)
                    
                    # 오디오 특징 추출
                    features = extract_features(audio_path)
                    
                    # 레이블 추가
                    X.append(features)
                    y.append(label)
    
    # X, y를 numpy 배열로 변환
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# 데이터 로딩
X, y = load_data(audio_dir)
print(f"Loaded {len(X)} samples")

# 레이블 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 데이터 차원 변경 (CNN에 맞게 3D 배열로 변경)
X = X[..., np.newaxis]  # (samples, height, width, 1) 형태로 변경

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# CNN 모델 정의
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # 클래스 수에 맞는 출력
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# 모델 저장 (새로운 음성파일을 분류할 때 사용할 모델)
model.save('parrot_speech_model.h5')

# 오디오 파일을 mel-spectrogram으로 변환하는 함수
def extract_features(audio_path, target_length=130):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_resized = librosa.util.fix_length(mel_spec, size=target_length, axis=-1)
    mel_spec_db = librosa.power_to_db(mel_spec_resized, ref=np.max)
    return mel_spec_db

# 새로운 오디오 파일을 분류하는 함수
def classify_new_audio(model, audio_path, label_encoder):
    # 특징 추출
    features = extract_features(audio_path)
    
    # 모델 입력 형태로 차원 변경 (1D -> 3D)
    features = features[..., np.newaxis]  # (height, width, 1)
    features = np.expand_dims(features, axis=0)  # (1, height, width, 1)
    
    # 예측
    prediction = model.predict(features)
    
    # 예측된 레이블을 원래 레이블로 변환
    predicted_label_idx = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_label_idx)
    
    return predicted_label[0]

# 모델 불러오기
model = tf.keras.models.load_model('parrot_speech_model.h5')

# 레이블 인코딩 불러오기
le = LabelEncoder()
le.fit(os.listdir(audio_dir))  # 레이블 인코딩을 위한 학습

# 새로운 음성파일 경로
new_audio_file = 'C:/Users/LG/Desktop/2025/구글 클라우드/hello.wav'

# 예측 실행
predicted_label = classify_new_audio(model, new_audio_file, le)

# 예측 결과 출력
print(f"The predicted label for the audio file is: {predicted_label}")
