import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

carpeta_csv = "vectores_csv"
output_folder = "graficos_tsne"
os.makedirs(output_folder, exist_ok=True)

for archivo in os.listdir(carpeta_csv):
    if archivo.endswith(".csv"):
        ruta_archivo = os.path.join(carpeta_csv, archivo)
        
        try:
            data = pd.read_csv(ruta_archivo)
            y = data.iloc[:, 0] # Etiqueta
            X = data.iloc[:, 1:] # Características
            tsne = TSNE(n_components=2, random_state=42) # Reducir a 2 dimensiones
            X_reduced = tsne.fit_transform(X)
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=5)
            plt.colorbar(scatter, label="Clase")
            plt.title(f"t-SNE de {archivo} (2D)")
            plt.xlabel("X")
            plt.ylabel("Y")
            nombre_grafico = os.path.splitext(archivo)[0] + "_tsne.png"
            plt.savefig(os.path.join(output_folder, nombre_grafico))
            plt.close()
            print(f"Gráfico guardado: {nombre_grafico}")

        except Exception as e:
            print(f"Error al procesar {archivo}: {e}")
    else:
        print(f"Error al procesar {archivo}, debe de estar en CSV.")

print("Proceso finalizado.")