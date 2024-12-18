from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

input_folders = ["data/bien", "data/mal"]
output_folders = ["espectrogramas/mel/bien", "espectrogramas/mel/mal"]

for folder in output_folders:
    os.makedirs(os.path.join(folder, "descriptores"), exist_ok=True)
    os.makedirs(os.path.join(folder, "solo"), exist_ok=True)

n_mels_range = range(12, 16)
window_duration = 0.02 # Ventana en segundos (20 ms)
overlap = 0.75 # Porcentaje de traslapo
window_type = 'hann'

for input_folder, output_folder in zip(input_folders, output_folders):
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
            audio, samplerate = sf.read(filepath)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1) # Convertir a mono promediando los canales
            
            # Calcular el tamaño de la ventana en muestras
            n_fft = int(window_duration * samplerate) # Número de puntos FFT
            hop_length = int(n_fft * (1 - overlap)) # Salto entre ventanas
            
            for n_mels in n_mels_range:
                descriptors_folder = os.path.join(output_folder, "descriptores", f"{n_mels}")
                solo_folder = os.path.join(output_folder, "solo", f"{n_mels}")
                os.makedirs(descriptors_folder, exist_ok=True)
                os.makedirs(solo_folder, exist_ok=True)
                descriptor_path = os.path.join(descriptors_folder, f"{os.path.splitext(filename)[0]}.png")
                solo_path = os.path.join(solo_folder, f"{os.path.splitext(filename)[0]}.png")
                
                if os.path.exists(descriptor_path) and os.path.exists(solo_path):
                    print(f"Ambos espectrogramas ya existen para {n_mels}, saltando: {output_folder}/{filename}")
                    continue
                
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio, sr=samplerate, n_fft=n_fft, hop_length=hop_length, 
                    window=window_type, n_mels=n_mels, power=2.0
                ) # Calcular el espectrograma MEL
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max) # Convertir a dB
                
                # Generar espectrograma con descriptores
                if not os.path.exists(descriptor_path):
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(
                        mel_spectrogram_db, sr=samplerate, hop_length=hop_length, 
                        x_axis="time", y_axis="mel", fmax=samplerate / 2, cmap="viridis"
                    )
                    plt.colorbar(label="Intensidad (dB)")
                    plt.title(f"Espectrograma MEL (n_mels={n_mels}) de {filename}")
                    plt.tight_layout()
                    plt.savefig(descriptor_path)
                    plt.close()
                    print(f"Espectrograma con descriptores guardado: {descriptor_path}")
                
                # Generar espectrograma sin descriptores
                if not os.path.exists(solo_path):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    librosa.display.specshow(
                        mel_spectrogram_db, sr=samplerate, hop_length=hop_length, 
                        x_axis="time", y_axis="mel", fmax=samplerate / 2, cmap="viridis", ax=ax
                    )
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    fig.savefig(solo_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    
                    with Image.open(solo_path) as img:
                        img_cropped = img.crop(img.getbbox())
                        img_cropped.save(solo_path)

                    print(f"Espectrograma sin descriptores guardado: {solo_path}")
        else:
            print(f"El archivo '{filename}' de '{input_folder}' debe de estar en .wav - No fue procesado")

print("Obtenido los espectrogramas MEL de todos los audios.")

# Copyright (c) 2024 MrMike92