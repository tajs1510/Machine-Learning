from flask import Flask, render_template

app = Flask(__name__)

# Định nghĩa các giá trị ma trận nhầm lẫn
TN = 50
FP = 10
FN = 5
TP = 30

# Tính toán các chỉ số và làm tròn đến 2 chữ số thập phân
accuracy = round((TP + TN) / (TP + TN + FP + FN), 2)
recall = round(TP / (TP + FN), 2)
specificity = round(TN / (TN + FP), 2)
precision = round(TP / (TP + FP), 2)
f1 = round(2 * precision * recall / (precision + recall), 2)

@app.route('/')
def home():
    # Truyền các giá trị vào template để hiển thị
    return render_template(
        'index.html',
        TN=TN,
        FP=FP,
        FN=FN,
        TP=TP,
        accuracy=accuracy,
        recall=recall,
        specificity=specificity,
        precision=precision,
        f1=f1
    )

if __name__ == '__main__':
    app.run(debug=True)
