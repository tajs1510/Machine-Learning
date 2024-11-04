import torch

def zScoreScaling(tensor):
    # Tính giá trị trung bình (mu) cho mỗi cột
    mu = []
    for j in range(tensor.size(1)):
        column_sum = torch.sum(tensor[:, j])
        mu.append(column_sum / tensor.size(0))

    mu = torch.tensor(mu)

    # Tính độ lệch chuẩn (sigma) cho mỗi cột
    sigma = []
    for j in range(tensor.size(1)):
        variance_sum = torch.sum((tensor[:, j] - mu[j]) ** 2)
        variance = variance_sum / tensor.size(0)
        sigma.append(variance.sqrt())

    sigma = torch.tensor(sigma)

    # Tính Z-score cho từng phần tử trong tensor
    z_scores = (tensor - mu) / sigma
    return z_scores

def minMaxScaling(tensor):
    # Tính giá trị nhỏ nhất và lớn nhất cho mỗi cột
    min_values = []
    max_values = []

    for j in range(tensor.size(1)):
        min_values.append(torch.min(tensor[:, j]))
        max_values.append(torch.max(tensor[:, j]))

    min_values = torch.tensor(min_values)
    max_values = torch.tensor(max_values)

    # Tính Min-Max Scaling cho từng phần tử trong tensor
    scaled_tensor = (tensor - min_values) / (max_values - min_values)
    return scaled_tensor

class Linear:
    def __init__(self, input_dim, output_dim):
        # Khởi tạo trọng số (weights) và độ chệch (bias)
        self.weight = torch.randn(input_dim, output_dim) * 0.01
        self.bias = torch.zeros(output_dim)

    def forward(self, x):
        # Dự đoán đầu ra dựa trên đầu vào
        return x @ self.weight + self.bias

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo tensor đầu vào
    tensor = torch.tensor([[1.0, 2.0, 3.0], 
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

    # Chuẩn hóa Z-score
    zscore = zScoreScaling(tensor)
    print("Z-score Scaling:")
    print(zscore)

    # Chuẩn hóa Min-Max
    min_max = minMaxScaling(tensor)
    print("\nMin-Max Scaling:")
    print(min_max)

    # Tạo tensor đầu vào cho lớp Linear
    input_tensor = torch.tensor([[1.0, 2.0, 3.0]])  # Nhập vào có kích thước (1, 3)

    # Khởi tạo mô hình Linear
    linear = Linear(input_dim=3, output_dim=2)
    out = linear.forward(input_tensor)
    
    print("\nLinear Model Output:")
    print(out)
    print(f"Weights:\n{linear.weight}")
    print(f"Bias:\n{linear.bias}")
