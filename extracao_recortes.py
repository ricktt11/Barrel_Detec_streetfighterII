import os
import cv2

frames_dir = "dataset/frame"      # Pasta dos frames
labels_dir = "dataset/labels"     # Pasta dos labels
output_dir = "dataset_fighter_cut"  # Pasta de saída

# Cria pastas de saída para cada classe
for folder in ("ken", "barris"):
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Itera pelos frames
for frame_name in os.listdir(frames_dir):
    if not frame_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Ignora arquivos não-imagem

    frame_path = os.path.join(frames_dir, frame_name)
    label_path = os.path.join(labels_dir, os.path.splitext(frame_name)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"[AVISO] Label não encontrada: {frame_name}")
        continue  # Pula se label não existe

    img = cv2.imread(frame_path)    # Carrega imagem
    h, w = img.shape[:2]            # Obtém altura e largura

    # Lê cada linha do arquivo de label
    with open(label_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Ignora linhas inválidas

            class_id, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])

            # Converte coordenadas normalizadas para pixels
            x1 = max(int((x - bw/2) * w), 0)
            y1 = max(int((y - bh/2) * h), 0)
            x2 = min(int((x + bw/2) * w), w)
            y2 = min(int((y + bh/2) * h), h)

            # Define caminho e nome do arquivo de saída
            save_path = os.path.join(output_dir, "ken" if class_id == 0 else "barris",
                                     f"{os.path.splitext(frame_name)[0]}_{i}.jpg")
            cv2.imwrite(save_path, img[y1:y2, x1:x2])  # Salva recorte
            print(f"[OK] Recorte salvo: {save_path}")

print("\nExtração concluída com sucesso!")
