import os
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

input_folder = "vectores_csv"
output_folder = "resultados"
os.makedirs(output_folder, exist_ok=True)

# Parámetros para el grid search
epochs_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
learning_rate_options = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
activation_options = [nn.ReLU, nn.Tanh, nn.Sigmoid]
regularization_options = [None, nn.Dropout, nn.BatchNorm1d, nn.LayerNorm]
regularization_rate_options = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
hidden_layers_options = [1, 2, 3]
neurons_per_layer_options = [64, 128, 256, 512]

# Combinaciones de hiperparámetros
param_grid = list(itertools.product(
    hidden_layers_options,
    neurons_per_layer_options,
    epochs_options,
    learning_rate_options,
    activation_options,
    regularization_options,
    regularization_rate_options
))

def load_existing_results(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        existing_combinations = set(
            tuple(row) for row in df[[
                "Capas ocultas",
                "Neuronas por capa",
                "Epocas",
                "Tasa de aprendizaje",
                "Activacion",
                "Regularizacion",
                "Tasa de regularizacion"
            ]].values
        )
        return existing_combinations
    else:
        return set()

# Modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, activation, regularization=None, regularization_rate=0.0):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, neurons_per_layer))
            layers.append(activation())
            if regularization == nn.Dropout:
                layers.append(regularization(p=regularization_rate))
            elif regularization in [nn.BatchNorm1d, nn.LayerNorm]:
                layers.append(regularization(neurons_per_layer))
            in_features = neurons_per_layer

        layers.append(nn.Linear(in_features, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def save_result_to_txt(result, file_path):
    with open(file_path, 'a') as f:
        f.write(str(result) + "\n")

for n_mels in range(12, 16):
    input_file = os.path.join(input_folder, f"{n_mels}_mels.csv")
    if not os.path.exists(input_file):
        print(f"Archivo no encontrado: {input_file}. Saltando...")
        continue

    print(f"Procesando {n_mels} mels...")
    df = pd.read_csv(input_file)
    X = df.iloc[:, 1:].values
    y = df["label"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    results_csv = os.path.join(output_folder, f"{n_mels}_mels_results.csv")
    existing_combinations = load_existing_results(results_csv)
    total_combinations = len(param_grid)

    if len(existing_combinations) >= total_combinations:
        print(f"Todas las combinaciones ya han sido evaluadas para {n_mels} mels. Saltando...")
        continue

    results = []

    for i, (hidden_layers, neurons_per_layer, epochs, lr, activation, regularization, reg_rate) in enumerate(param_grid):
        combination = (
            hidden_layers, neurons_per_layer, epochs, lr, activation.__name__,
            regularization.__name__ if regularization else "Ninguno", reg_rate
        )

        if combination in existing_combinations:
            print(f"Combinación {i+1}/{total_combinations} ya evaluada para {n_mels} mels. Saltando...")
            continue

        print(f"Evaluando combinación {i+1}/{total_combinations} para {n_mels} mels: "
              f"hidden_layers={hidden_layers}, neurons_per_layer={neurons_per_layer}, epochs={epochs}, lr={lr}, "
              f"activation={activation.__name__}, regularization={regularization}, reg_rate={reg_rate}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(
            input_size=X_train.shape[1],
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons_per_layer,
            activation=activation,
            regularization=regularization,
            regularization_rate=reg_rate
        )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Entrenamiento
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            X_train_tensor_device = X_train_tensor.to(device)
            y_train_tensor_device = y_train_tensor.to(device)
            outputs = model(X_train_tensor_device)
            loss = criterion(outputs, y_train_tensor_device)
            loss.backward()
            optimizer.step()

        # Evaluación
        model.eval()
        with torch.no_grad():
            X_test_tensor_device = X_test_tensor.to(device)
            y_test_tensor_device = y_test_tensor.to(device)
            y_pred = model(X_test_tensor_device).argmax(dim=1).cpu().numpy()
            accuracy = accuracy_score(y_test, y_pred)

        result = {
            "Capas ocultas": hidden_layers,
            "Neuronas por capa": neurons_per_layer,
            "Epocas": epochs,
            "Tasa de aprendizaje": lr,
            "Activacion": activation.__name__,
            "Regularizacion": regularization.__name__ if regularization else "Ninguno",
            "Tasa de regularizacion": reg_rate,
            "Presicion": accuracy
        }
        results.append(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        print(f"Resultados guardados para {n_mels} mels en {results_csv}")

    print(f"Se termino de analizar {n_mels}")

print("Se analizaron todos los mels.")