import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def principal():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    transformacion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    datos_entrenamiento = datasets.FashionMNIST(root='./datos', train=True, download=True, transform=transformacion)
    cargador_entrenamiento = torch.utils.data.DataLoader(datos_entrenamiento, batch_size=64, shuffle=True)

    datos_prueba = datasets.FashionMNIST(root='./datos', train=False, download=True, transform=transformacion)
    cargador_prueba = torch.utils.data.DataLoader(datos_prueba, batch_size=64, shuffle=False)

    class RedNeuronal(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.dropout(F.relu(self.fc3(x)))
            x = self.fc4(x)
            return F.log_softmax(x, dim=1)

    modelo = RedNeuronal().to(dispositivo)
    funcion_perdida = nn.NLLLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.003)

    epocas = 5
    for epoca in range(epocas):
        perdida_entrenamiento = 0
        modelo.train()
        for imagenes, etiquetas in cargador_entrenamiento:
            imagenes = imagenes.view(imagenes.shape[0], -1).to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            optimizador.zero_grad()
            salida = modelo(imagenes)
            perdida = funcion_perdida(salida, etiquetas)
            perdida.backward()
            optimizador.step()

            perdida_entrenamiento += perdida.item()

        modelo.eval()
        precision = 0
        perdida_prueba = 0
        with torch.no_grad():
            for imagenes, etiquetas in cargador_prueba:
                imagenes = imagenes.view(imagenes.shape[0], -1).to(dispositivo)
                etiquetas = etiquetas.to(dispositivo)
                salida_log = modelo(imagenes)
                perdida_prueba += funcion_perdida(salida_log, etiquetas)

                probabilidades = torch.exp(salida_log)
                _, prediccion = probabilidades.topk(1, dim=1)
                iguales = prediccion == etiquetas.view(*prediccion.shape)
                precision += torch.mean(iguales.type(torch.FloatTensor)).item()

        print(f"Época {epoca+1}/{epocas}.. "
              f"Pérdida entrenamiento: {perdida_entrenamiento/len(cargador_entrenamiento):.3f}.. "
              f"Pérdida prueba: {perdida_prueba/len(cargador_prueba):.3f}.. "
              f"Precisión prueba: {precision/len(cargador_prueba):.3f}")

    def visualizar_prediccion(imagen, probabilidades, clases):
        probabilidades = probabilidades.cpu().numpy().squeeze()
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 4), ncols=2)
        ax1.imshow(imagen.numpy().squeeze(), cmap='gray')
        ax1.axis('off')
        ax2.barh(np.arange(10), probabilidades)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(clases)
        ax2.set_title('Probabilidades')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()

    modelo.eval()
    with torch.no_grad():
        iterador = iter(cargador_prueba)
        imagenes, etiquetas = next(iterador)
        imagen = imagenes[0]
        etiqueta_real = etiquetas[0]

        imagen_procesada = imagen.view(1, 784).to(dispositivo)
        salida = modelo(imagen_procesada)
        probabilidades = torch.exp(salida)

        clases = datos_entrenamiento.classes
        print("Etiqueta real: ", clases[etiqueta_real])
        print("Predicción:    ", clases[probabilidades.argmax().item()])
        visualizar_prediccion(imagen, probabilidades, clases)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    principal()
