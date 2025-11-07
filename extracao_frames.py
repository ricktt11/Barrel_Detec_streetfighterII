import cv2  # OpenCV para vídeo e imagem
import os   # Para criar diretórios

video_path = 'streetfighter.mp4'  # Vídeo de entrada
output_dir = 'frame'               # Pasta para salvar frames
os.makedirs(output_dir, exist_ok=True)  # Cria pasta se não existir

cap = cv2.VideoCapture(video_path)  # Abre o vídeo
frame_rate = 5  # Salva 1 frame a cada 5
count = 0       # Contador de frames

while cap.isOpened():
    ret, frame = cap.read()  # Lê o frame
    if not ret:
        break
    if count % frame_rate == 0:  # Salva frame a cada 5
        cv2.imwrite(f"{output_dir}/frame_{count}.jpg", frame)  # Salva frame
    count += 1

cap.release()  # Libera vídeo
