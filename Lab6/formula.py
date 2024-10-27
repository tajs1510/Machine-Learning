import torch
import torch.nn.functional as F

def crossEntropyLoss(output, target):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    return loss.item()

# Công thức tính Mean Square Error
def meanSquareError(output, target):
    return torch.sum((output - target) ** 2) 

# Công thức tính BinaryEntropy Loss
def binaryEntropyLoss(output, target, n):
    # Sử dụng công thức Binary Cross Entropy Loss: 
    # -1/n * Σ [target_i * log(output_i) + (1 - target_i) * log(1 - output_i)]
    epsilon = 1e-9  # Để tránh log(0)
    output = torch.clamp(output, min=epsilon, max=1 - epsilon)  # Đảm bảo giá trị nằm trong khoảng (0, 1)
    return (-1 / n) * torch.sum(target * torch.log(output) + (1 - target) * torch.log(1 - output))

# Đầu vào và mục tiêu
inputs = torch.tensor([0.1, 0.3, 0.6, 0.7])
target = torch.tensor([0.31, 0.32, 0.8, 0.2])
n = len(inputs)

# Tính toán các giá trị lỗi
mse = meanSquareError(inputs, target)
binary_loss = binaryEntropyLoss(inputs, target, n)
cross_loss = crossEntropyLoss(inputs, target)

# In ra kết quả
print(f"Mean Square Error: {mse}")
print(f"Binary Entropy Loss: {binary_loss}")
print(f"Cross Entropy Loss: {cross_loss}")

def sigmoid(x: torch.tensor) -> torch.tensor:
    return 1 / (1 + torch.exp(-x))

# Công thức hàm relu
def relu(x: torch.tensor) -> torch.tensor:
    return torch.maximum(x, torch.tensor(0.0))

# Công thức hàm softmax
def softmax(zi: torch.tensor) -> torch.tensor:
    exp_zi = torch.exp(zi - torch.max(zi))  # Để tránh tràn số
    return exp_zi / exp_zi.sum(dim=0)

# Công thức hàm tanh
def tanh(x: torch.tensor) -> torch.tensor:
    return torch.tanh(x)

# Tạo tensor đầu vào
x = torch.tensor([1, 5, -4, 3, -2], dtype=torch.float32)

# Tính toán giá trị của các hàm
f_sigmoid = sigmoid(x)
f_relu = relu(x)
f_softmax = softmax(x)
f_tanh = tanh(x)

# In kết quả
print(f"Sigmoid = {f_sigmoid}")
print(f"ReLU = {f_relu}")
print(f"Softmax = {f_softmax}")
print(f"Tanh = {f_tanh}")