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

data_file = "embeddings_audios.csv"
df = pd.read_csv(data_file)
X = df.iloc[:, 1:].values
y = df["label"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

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

# Modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, activation, regularization=None, regularization_rate=0.0):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        # Capas ocultas
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, neurons_per_layer))
            layers.append(activation())
            if regularization == nn.Dropout:
                layers.append(regularization(p=regularization_rate))
            elif regularization in [nn.BatchNorm1d, nn.LayerNorm]:
                layers.append(regularization(neurons_per_layer))
            in_features = neurons_per_layer

        layers.append(nn.Linear(neurons_per_layer, 2)) # Capa de salida
        self.model = nn.Sequential(*layers) # Capas en secuencia

    def forward(self, x):
        return self.model(x)

# Verificar si los resultados ya existen
def load_existing_results(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=["Capas ocultas", "Neuronas por capa", "Epocas", "Tasa de aprendizaje", "Activacion", "Regularizacion", "Tasa de regularizacion", "Presicion"])

def save_results(results, file_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_path, index=False)

# Cargar resultados si es que existen
results_csv = "resultados/resultados_embeddings_audios.csv"
existing_results = load_existing_results(results_csv)
existing_combinations = set([tuple(x) for x in existing_results[["Capas ocultas", "Neuronas por capa", "Epocas", "Tasa de aprendizaje", "Activacion", "Regularizacion", "Tasa de regularizacion"]].values])
results = []

# Realizar búsqueda en cuadrícula
for i, (hidden_layers, neurons_per_layer, epochs, lr, activation, regularization, reg_rate) in enumerate(param_grid):
    combination = (
        hidden_layers, neurons_per_layer, epochs, lr, activation.__name__, regularization.__name__ if regularization else "None", reg_rate
    )
    if combination in existing_combinations:
        print(f"Combinación {i+1}/{len(param_grid)} ya evaluada, saltando...")
        continue

    print(f"Evaluando combinación {i+1}/{len(param_grid)}: "
          f"epochs={epochs}, lr={lr}, activation={activation.__name__}, "
          f"regularization={regularization}, reg_rate={reg_rate}, "
          f"hidden_layers={hidden_layers}, neurons_per_layer={neurons_per_layer}")

    # Crear modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        input_size=X_train.shape[1],
        activation=activation,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        regularization=regularization,
        regularization_rate=reg_rate
    ).to(device)
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
    print(f"Combinación {i+1} - Accuracy: {accuracy:.4f}")
    save_results(results, results_csv)

print(f"Resultados guardados en {results_csv}")