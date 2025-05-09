import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('CIFAR10_data', download=True, train=True, transform=transform)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10) 
        
    def forward(self, x):
        x = x.view(x.shape[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
print_every = 40
steps = 0

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print(f"Epoch: {e+1}/{epochs}... "
                  f"Loss: {running_loss/print_every:.4f}")
            running_loss = 0
def view_classify(img, ps):
    import matplotlib.pyplot as plt
    import numpy as np
    
    ps = ps.data.cpu().numpy().squeeze()
    img = img.cpu().numpy().transpose(1, 2, 0)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(np.clip((img * 0.5 + 0.5), 0, 1)) 
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
images, labels = next(iter(trainloader))
img = images[0].view(1, 32*32*3).to(device)
with torch.no_grad():
    logits = model(img)
    
ps = F.softmax(logits, dim=1)
view_classify(images[0], ps)