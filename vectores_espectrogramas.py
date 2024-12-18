import os
import numpy as np
import pandas as pd
from PIL import Image

input_folders = {
    "espectrogramas/mel/bien/solo": 0,
    "espectrogramas/mel/mal/solo": 1
}
output_folder = "vectores_csv"
os.makedirs(output_folder, exist_ok=True)
data_by_mels = {}
target_size = (16, 8)

for folder, label in input_folders.items():
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".png"):
                    filepath = os.path.join(subfolder_path, filename)

                    with Image.open(filepath) as img:
                        img_resized = img.resize(target_size)
                        img_array = np.array(img_resized).flatten()

                        if subfolder not in data_by_mels:
                            data_by_mels[subfolder] = {"data": [], "labels": []}

                        data_by_mels[subfolder]["data"].append(img_array)
                        data_by_mels[subfolder]["labels"].append(label)

for n_mels, content in data_by_mels.items():
    data = np.array(content["data"])
    labels = np.array(content["labels"]).reshape(-1, 1)
    combined = np.hstack((labels, data))
    columns = ["label"] + [str(i) for i in range(data.shape[1])]
    df = pd.DataFrame(combined, columns=columns)
    output_csv = os.path.join(output_folder, f"{n_mels}_mels.csv")
    df.to_csv(output_csv, index=False)
    print(f"Archivo CSV generado: {output_csv}")

print("Todos los vectores característicos se han generado.")

# Copyright (c) 2024 MrMike92