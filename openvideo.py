import cv2, os, joblib, numpy as np

# Configurações
VIDEO_PATH = "streetfighter.mp4"      # Vídeo de entrada
LABELS_DIR = "dataset/labels"         # Labels de cada frame
MLP_PATH, SCALER_PATH = "mlp_model.pkl", "scaler.pkl"  # Modelo e scaler
IMG_SIZE, FRAME_SIZE = (64,64), (1200,720)  # Tamanho para MLP e exibição

# Carrega modelo e scaler
mlp, scaler = joblib.load(MLP_PATH), joblib.load(SCALER_PATH)

# Função de predição
def predict_crop(img):
    # Redimensiona e normaliza a imagem
    img_scaled = scaler.transform(cv2.resize(img, IMG_SIZE).flatten().reshape(1,-1)/255.0)
    pred = mlp.predict(img_scaled)[0]            # Classe prevista
    prob = mlp.predict_proba(img_scaled)[0][pred] # Probabilidade da classe
    return pred, prob

# Inicializa vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)      # Delay para mostrar vídeo no tempo correto
frame_idx, last_boxes = 0, []  # Contador e últimas caixas detectadas

while True:
    ret, frame = cap.read()   # Lê frame
    if not ret: break          # Sai se acabou
    frame = cv2.resize(frame, FRAME_SIZE)
    h, w = frame.shape[:2]
    boxes = []                 # Lista de detecções do frame

    # Lê labels do frame se existir
    label_file = os.path.join(LABELS_DIR, f"frame_{frame_idx}.txt")
    if os.path.exists(label_file):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)!=5: continue
                cls, xc, yc, bw, bh = map(float, parts)

                # Converte coordenadas normalizadas para pixels
                x1, y1 = max(int((xc-bw/2)*w),0), max(int((yc-bh/2)*h),0)
                x2, y2 = min(int((xc+bw/2)*w),w), min(int((yc+bh/2)*h),h)

                crop = frame[y1:y2, x1:x2]  # Recorta região do objeto
                if crop.size==0: continue    # Ignora crops vazios

                pred, prob = predict_crop(crop)  # Predição MLP
                color, text = ((0,255,0),"Ken") if pred==0 else ((0,0,255),"Barril")
                boxes.append((x1,y1,x2,y2,color,text,prob))
        last_boxes = boxes          # Guarda para próximo frame
    else:
        boxes = last_boxes          # Usa boxes anteriores se não houver label

    # Desenha caixas e texto
    for x1,y1,x2,y2,color,text,prob in boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)           # Desenha retângulo
        cv2.putText(frame,f"{text} {prob:.2f}",(x1,y1-5),     # Escreve texto
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

    cv2.imshow("Detecção MLP", frame)  # Mostra frame
    if cv2.waitKey(delay) & 0xFF == ord('q'): break  # Sai se apertar 'q'
    frame_idx += 1

# Finaliza vídeo
cap.release()              
cv2.destroyAllWindows()
