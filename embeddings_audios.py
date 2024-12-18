import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

input_folders = {"data/bien": 0, "data/mal": 1}
output_csv = "embeddings_audios.csv"
embedding_size = 512
target_sample_rate = 16000 # Frecuencia de muestreo requerida por el modelo (estricta a 16 kHz)
model_name = "microsoft/wavlm-base-plus" # Cargar modelo WavLM
device = "cuda" if torch.cuda.is_available() else "cpu" # Cargar feature extractor
model = WavLMModel.from_pretrained(model_name).to(device)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
data = []
labels = []

for folder, label in input_folders.items():
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder, filename)
            waveform, sample_rate = torchaudio.load(filepath)

            if waveform.shape[0] > 1: # Convertir a mono si es estéreo
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resamplear a 16 kHz
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)

            audio = waveform.squeeze().numpy()
            audio = audio / np.max(np.abs(audio)) # Normalizar el audio
            inputs = feature_extractor(audio, sampling_rate=target_sample_rate, return_tensors="pt", padding=True) # Extraer embeddings
            inputs = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            data.append(embeddings)
            labels.append(label)

data = np.array(data)
pca = PCA(n_components=embedding_size)
data_reduced = pca.fit_transform(data)
labels = np.array(labels).reshape(-1, 1)
combined = np.hstack((labels, data_reduced))
columns = ["label"] + [f"embedding_{i}" for i in range(embedding_size)]
df = pd.DataFrame(combined, columns=columns)
df.to_csv(output_csv, index=False)
print(f"Archivo CSV con PCA generado: {output_csv}")

# Copyright (c) 2024 MrMike92