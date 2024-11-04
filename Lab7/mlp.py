import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

# Thiết lập thiết bị
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Tải dữ liệu CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])


trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


def imshow(img):
    img = img / 2 + 0.5  # Bỏ chuẩn hóa để ảnh về đúng màu
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Lấy và hiển thị 5 ảnh đầu tiên từ tập dữ liệu test
for i, (images, labels) in enumerate(testloader, 0):
    # Hiển thị 5 ảnh đầu tiên
    for j in range(5):
        print(f"Label: {labels[j].item()}")  # In nhãn của ảnh
        imshow(images[j])  # Hiển thị ảnh
    break  # Dừng sau khi hiển thị 5 ảnh đầu tiên

def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),  # Làm phẳng ảnh từ 3D thành 1D
        nn.Linear(n_features, 512),  # Lớp fully connected đầu tiên
        nn.ReLU(),                   # Hàm kích hoạt ReLU
        nn.Linear(512, 256),         # Lớp fully connected thứ hai
        nn.ReLU(),
        nn.Linear(256, 128),         # Lớp fully connected thứ ba
        nn.ReLU(),
        nn.Linear(128, 10)           # Lớp output với 10 đầu ra tương ứng 10 lớp của CIFAR-10
    )
    return model

# Số lượng đầu vào (tương ứng với số lượng đặc trưng của một ảnh 32x32x3)
n_features = 32 * 32 * 3
model = getModel(n_features)

# Learning rate
lr = 0.01

# Khởi tạo optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Khởi tạo hàm loss function
loss_fn = nn.CrossEntropyLoss()

# Hiển thị model
model

def evaluate(model, testloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = test_loss / len(testloader)
    return test_loss, accuracy

n_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        
        inputs, labels = inputs.to(device), labels.to(device)



        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()


        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
     

    epoch_accuracy = 100 * running_correct / total
    epoch_loss = running_loss / (i + 1)
    test_loss, test_accuracy = evaluate(model, testloader, loss_fn)
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)