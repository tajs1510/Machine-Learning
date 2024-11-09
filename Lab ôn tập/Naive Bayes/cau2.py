import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Dữ liệu thực tế và dự đoán (ví dụ)
true_labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
predicted_labels = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0]

# Tính toán ma trận nhầm lẫn
cm = confusion_matrix(true_labels, predicted_labels)

# Vẽ ma trận nhầm lẫn sử dụng seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Tính các chỉ số đánh giá
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / np.sum(cm)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# In ra các chỉ số đánh giá
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
