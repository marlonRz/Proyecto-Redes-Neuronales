import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True
    )

    model = CIFAR10Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_path = "cifar10_model.pth"

    if not os.path.exists(model_path):
        print("\n🔄 Entrenando el modelo...\n")
        epochs = 5
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"✅ Époch {epoch+1}/{epochs}, Pérdida: {running_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), model_path)
        print("\n💾 Modelo guardado exitosamente en 'cifar10_model.pth'\n")
    else:
        print("\n📂 Modelo ya guardado, saltando entrenamiento.\n")

    print("\n📥 Cargando el modelo guardado...\n")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_iter = iter(test_loader)
    image, label = next(data_iter)
    image = image.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    image_cpu = image.cpu().numpy().squeeze().transpose((1, 2, 0))
    image_cpu = (image_cpu * 0.5) + 0.5

    classes = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

    plt.imshow(image_cpu)
    plt.title(f'🎯 Etiqueta Real: {classes[label.item()]} | 🧠 Predicción: {classes[predicted.item()]}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()