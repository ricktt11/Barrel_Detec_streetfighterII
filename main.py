import os, cv2, numpy as np, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Configurações
base_dir = "dataset_fighter_cut"
folders = {"ken": 0, "barris": 1}  # nome: label
img_size = (64, 64)

# Carrega imagens e rótulos
def load_images(folder, label, size):
    data, labels = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                img = cv2.resize(img, size).flatten() / 255.0
                data.append(img)
                labels.append(label)
    return data, labels

X, y = [], []
for name, label in folders.items():
    d, l = load_images(os.path.join(base_dir, name), label, img_size)
    X += d
    y += l

X = np.array(X)
y = np.array(y)

# Divide treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normaliza standardscaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Treina modelo
mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42)
print("Treinando modelo...")
mlp.fit(X_train, y_train)

# Avaliação no conjunto de teste
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\nAcurácia de teste: {acc:.2f}%")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=list(folders.keys()), digits=4))

# Validação cruzada (5 divisões)
print("\nValidação cruzada (5 folds):")
scores = cross_val_score(mlp, X, y, cv=5)
print("Acurácias individuais:", np.round(scores*100, 2))
print("Média geral:", np.mean(scores)*100)

# Salva modelo e scaler
joblib.dump(mlp, "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModelo e scaler salvos com sucesso!")
