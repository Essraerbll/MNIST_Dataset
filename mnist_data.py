import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Veri Dönüşümleri ve DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1000, shuffle=False
)

# Cihaz seçimi (CUDA varsa GPU, yoksa CPU kullanılır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Tanımı
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)

# Modeli ve kriteri cihaza taşı
model = Net().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss().to(device)

# Eğitim ve Test Fonksiyonları
def train(epoch):
    model.train()
    total_loss, correct = 0, 0
    for data, target in train_loader:
        # Verileri cihaza taşı
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        correct += (output.argmax(1) == target).sum().item()
    
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def test():
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            # Verileri cihaza taşı
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            correct += (output.argmax(1) == target).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return avg_loss, accuracy

# Erken Durdurma Parametreleri
early_stopping_patience = 3  # Kaç epoch boyunca iyileşme olmazsa durdurulacak
best_loss = float('inf')  # En iyi (minimum) doğrulama kaybı
patience_counter = 0  # Sabit kalan epoch sayacı

# Eğitim Döngüsü
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(1, 101):  # Maksimum 100 epoch
    print(f"Epoch {epoch}")
    t_loss, t_acc = train(epoch)
    v_loss, v_acc = test()

    train_losses.append(t_loss)
    train_accuracies.append(t_acc)
    test_losses.append(v_loss)
    test_accuracies.append(v_acc)

    # Erken durdurma kontrolü
    if v_loss < best_loss:
        best_loss = v_loss
        patience_counter = 0  # İyileşme varsa sayaç sıfırlanır
        torch.save(model.state_dict(), 'best_model.pth')  # En iyi modeli kaydet
    else:
        patience_counter += 1  # İyileşme yoksa sayaç artırılır

    if patience_counter >= early_stopping_patience:
        print("Erken durdurma tetiklendi.")
        break

# En iyi modeli yükle
model.load_state_dict(torch.load('best_model.pth'))

# Grafik Çizimi
def plot_metrics(train_vals, test_vals, ylabel, title):
    plt.figure()
    plt.plot(train_vals, label='Train')
    plt.plot(test_vals, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

plot_metrics(train_losses, test_losses, 'Loss', 'Loss Over Epochs')
plot_metrics(train_accuracies, test_accuracies, 'Accuracy (%)', 'Accuracy Over Epochs')

# Test Sonuçlarını Görselleştir
model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    # Verileri cihaza taşı
    data, target = data.to(device), target.to(device)
    output = model(data)
    preds = output.argmax(1)

fig = plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data[i][0].cpu(), cmap='gray')
    plt.title(f"Pred: {preds[i].item()}\nTrue: {target[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Model Kaydet / Yükle
torch.save(model.state_dict(), 'mnist_cnn.pth')
loaded_model = Net().to(device)
loaded_model.load_state_dict(torch.load('mnist_cnn.pth'))
loaded_model.eval()
