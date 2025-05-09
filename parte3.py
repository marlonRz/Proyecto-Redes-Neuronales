import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuración básica
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para CIFAR-10 (las imágenes son 32x32 RGB)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Descargar y cargar los datos
train_data = datasets.CIFAR10('CIFAR10_data', download=True, train=True, transform=transform)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

# Definir la red neuronal
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 512)  # CIFAR-10: 32x32 imágenes RGB (3 canales)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # 10 clases en CIFAR-10
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Aplanar la imagen
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Logits, no aplicamos softmax aquí
        return x

model = Net().to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

# Entrenamiento
epochs = 5
print_every = 40
steps = 0

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        # Limpiar los gradientes
        optimizer.zero_grad()
        
        # Pase hacia adelante y hacia atrás
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print(f"Epoch: {e+1}/{epochs}... "
                  f"Loss: {running_loss/print_every:.4f}")
            running_loss = 0

# Función para visualizar predicciones (similar a helper.view_classify)
def view_classify(img, ps):
    import matplotlib.pyplot as plt
    import numpy as np
    
    ps = ps.data.cpu().numpy().squeeze()
    img = img.cpu().numpy().transpose(1, 2, 0)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(np.clip((img * 0.5 + 0.5), 0, 1))  # Desnormalizar
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['avión', 'auto', 'pájaro', 'gato', 'ciervo',
                        'perro', 'rana', 'caballo', 'barco', 'camión'])
    ax2.set_title('Probabilidad de clase')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

# Probar la red con una imagen
images, labels = next(iter(trainloader))
img = images[0].view(1, 32*32*3).to(device)

# Desactivar gradientes para la inferencia
with torch.no_grad():
    logits = model(img)
    
# Calcular probabilidades con softmax
ps = F.softmax(logits, dim=1)
view_classify(images[0], ps)