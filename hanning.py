import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram

input_folders = ["data/bien", "data/mal"]
output_folders = ["espectrogramas/hanning/bien", "espectrogramas/hanning/mal"]

for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

window_duration = 0.02 # Ventana en segundos (20 ms)
overlap = 0.75 # Porcentaje de traslapo
window_type = 'hann'

for input_folder, output_folder in zip(input_folders, output_folders):
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")

            if os.path.exists(output_path):
                print(f"Su epectrograma ya existe, saltando: {output_path}")
                continue

            audio, samplerate = sf.read(filepath)
            
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1) # Convertir a mono promediando los canales
            
            # Calcular el tamaño de la ventana en muestras
            nperseg = int(window_duration * samplerate)
            noverlap = int(nperseg * overlap)
            
            # Calcular el espectrograma
            f, t, Sxx = spectrogram(audio, fs=samplerate, window=window_type, 
                                    nperseg=nperseg, noverlap=noverlap, scaling='density')
            
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Convertir a dB y agregar 1e-10 para evitar log(0)
            plt.figure(figsize=(10, 4))
            plt.pcolormesh(t, f, Sxx_dB, shading='gouraud')
            plt.colorbar(label='Intensidad (dB)')
            plt.ylabel('Frecuencia (Hz)')
            plt.xlabel('Tiempo (s)')
            plt.title(f'Espectrograma de {filename}')
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"{output_path} - Guardado")

        else:
            print(f"El archivo '{filename}' de '{input_folder}' debe de estar en .wav - No fue procesado");

print("Obtenido los espectrogramas Hann de todos los audios.")