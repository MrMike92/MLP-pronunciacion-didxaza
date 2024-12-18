import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

archivo = 'embeddings_audios.csv'
output_folder = "graficos_tsne"
os.makedirs(output_folder, exist_ok=True)
data = pd.read_csv(archivo)
y = data.iloc[:, 0] # Etiqueta
X = data.iloc[:, 1:] # Características
tsne = TSNE(n_components=2, random_state=42) # Reducir a 2 dimensiones
X_reduced = tsne.fit_transform(X)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=5)
plt.colorbar(scatter, label="Clase")
plt.title("t-SNE de embeddings (2D)")
plt.xlabel("X")
plt.ylabel("Y")
nombre_grafico = os.path.splitext(archivo)[0] + ".png"
plt.savefig(os.path.join(output_folder, nombre_grafico))
plt.close()
print(f"Gráfico guardado: {nombre_grafico}")