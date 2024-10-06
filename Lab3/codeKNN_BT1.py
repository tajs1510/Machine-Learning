import pandas as pd
import numpy as np

# tạo hàm lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    '''Hàm này dùng để tải dữ liệu từ một tệp CSV'''
    try:
        # Sử dụng pandas để đọc file CSV và trả về một DataFrame
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Tệp '{filename}' không tồn tại.")
        return pd.DataFrame()  # Trả về một DataFrame trống nếu tệp không tồn tại
    except pd.errors.EmptyDataError:
        print("Tệp trống hoặc không có dữ liệu hợp lệ.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return pd.DataFrame()
def splitTrainTest(data, ratio_test):
    np.random.seed(28)  # không thay đổi mỗi lần chạy
    index_permu = np.random.permutation(len(data))  # xáo trộn index
    data_permu = data.iloc[index_permu]  # lấy lại dữ liệu tương ứng với index xáo trộn
    len_test = int(len(data_permu) * ratio_test)  # kích cỡ tập test
    test_set = data_permu.iloc[:len_test, :]  # tập test lấy phần đầu
    train_set = data_permu.iloc[len_test:, :]  # tập train lấy phần còn lại

    # Chia tập dữ liệu thành (X_train, y_train: Lấy tất cả cột trừ cột cuối), (X_test, y_test: Chỉ lấy cột cuối)
    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]

    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]

    return X_train, y_train, X_test, y_test
def get_words_frequency(data_X):  # tạo hàm lấy tần số từ, data_X --> DataFrame
    bag_words = np.concatenate([i[0].split(' ') for i in data_X.values], axis = None)
    # B1: lấy các giá trị của DataFrame --> Array
    # B2: lặp qua các phần tử trong mảng chính là i --> string (text)
    # B3: lấy các từ trong đoạn text ra --> [['VKE', 'đánh', 'CKTG', 'vậy', 'là', 'hay', 'rồi'], ['Đã', 'quá', 'VKE', 'ơi']]
    # B4: hàm concatenate với tham số axis = None có chức năng làm phẳng và đưa về mảng một chiều
    # --> array['VKE', 'đánh', 'vậy', 'là', 'hay', 'rồi', 'Đã', 'quá', 'VKE', 'ơi']

    bag_words = np.unique(bag_words) # loại bỏ các giá trị trùng và lấy giá trị duy nhất trong mảng bag_words
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype = int) # tạo ma trận 0 có kích cỡ [số dòng data_X(dòng) x số từ trong túi từ(cột)]

    word_freq = pd.DataFrame(matrix_freq, columns = bag_words) # tạo frame với matrix_freq, cột là các từ trong túi từ
    for id, text in enumerate(X_train.values.reshape(-1)):
    # hàm enumerate sẽ gán index cho mỗi phần từ, có tham số start để có thể điều chỉnh số bắt đầu
    # index lưu vào id, phần tử lưu vào text
        for j in bag_words: # đối với mỗi id (dòng), ta lặp qua các từ trong túi (cột)
            word_freq.at[id, j] = text.split(' ').count(j) # đếm từ đó có trong biến text và gán tại vị trí [id, j]
    return word_freq, bag_words # trả lại biến tần số từ, --> DataFrame(cột là các từ trong túi từ)
def transform(data_test, bags): # bags là bag_words được return từ hàm get_words_frequency, data_test dạng frame
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype = int)
    frame_0 = pd.DataFrame(matrix_0, columns = bags)
    for id, text in enumerate(data_test.values.reshape(-1)):
        print(text)
        for j in bags:
            frame_0.at[id, j] = text.split(' ').count(j)
    return frame_0
def cosine_distance(train_X_number_arr, test_X_number_arr):
    dict_kq = dict() # tạo dictionary trống
    for id, arr_test in enumerate(test_X_number_arr, start = 1):
    # tương tự lấy index cho mỗi phần tử trong mảng test_X_number_arr, index đánh bắt đầu bằng 1

        q_i = np.sqrt(sum(arr_test**2)) # căn của tổng ([q_i]^2), dùng để tính mẫu
        for j in train_X_number_arr:
            _tu = sum(j*arr_test) # tính tử: tổng q[i]*dj[i]

            # tính mẫu: (căn của tổng (q[i]^2)*(căn của tổng (dj[i])^2)
            d_j = np.sqrt(sum(j**2))
            _mau = d_j*q_i

            # kết quả: lấy tử chia mẫu --> khoảng cách của mỗi dòng trong test_X với các dòng trong train_X
            kq = _tu/_mau

            # nếu index có trong dict_kq rồi thì ta thêm giá trị kq vào, nếu chưa thì ta tạo khoá id với giá trị kq.
            if id in dict_kq:
                dict_kq[id].append(kq)
            else:
                dict_kq[id] = [kq]

    return dict_kq # --> Dictionary với key: dòng trong tập test, value: các giá trị đã được tính khoảng cách với các dòng trong tập train
    # ví dụ: {1: [2, 3, 4, 5, 6]}, 1 là dòng thứ nhất trong tập test, [2, 3, 4, 5, 6] là khoảng cách của dòng 1 trong tập test đến các dòng trong tập train
class KNNText:
    # hàm tạo
    def __init__(self, k):
          self.k = k  # Số lượng điểm lân cận gần nhất
          self.X_train = None
          self.y_train = None
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        self.X_test = X_test

        _distance = cosine_distance(self.X_train, self.X_test)

        # Reset the index of y_train (optional, depends on how it's stored)
        self.y_train.index = range(len(self.y_train))

        # Step 2: Combine distances with corresponding targets
        _distance_frame = pd.concat([pd.DataFrame(_distance), pd.DataFrame(self.y_train, columns=['target'])], axis=1)

        # Step 3: Prepare to store predictions
        target_predict = dict()

        # Step 4: For each test instance
        for i in range(1, len(self.X_test) + 1):
            # Get the distance values for the current test point and corresponding targets, sort by distance
            sorted_frame = _distance_frame[[i, 'target']].sort_values(by=i)

            # Get the top k nearest neighbors
            top_k = sorted_frame.head(self.k)

            # Count the most frequent target among the k neighbors
            predicted_target = top_k['target'].mode()[0]  # Majority vote

            # Add the result to the target_predict dictionary
            target_predict[i] = predicted_target

        return target_predict
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = 0

        for i, pred in predictions.items():
            if pred == y_test[i]:
                correct += 1

        return correct / len(y_test)  # Accuracy
    def cosine_distance(X_train, X_test):
    # Implement cosine similarity/distance calculation here
       pass

data = loadCsv('Education.csv')
# loại bỏ các kí tự đặc biệt
data['Text'] = data['Text'].apply(lambda x: x.replace(',', ''))
data['Text'] = data['Text'].apply(lambda x: x.replace('.', ''))
data['Text'][1]

X_train, y_train, X_test, y_test = splitTrainTest(data, 0.25)
print(len(X_train))
print(len(X_test))

words_train_fre, bags = get_words_frequency(X_train)
print(bags)
print(len(bags))
print(words_train_fre)

words_train_fre

words_test_fre = transform(X_test, bags)

words_test_fre

knn = KNNText(k = 2)
knn.fit(words_train_fre.values, y_train)
pred_ =  pd.DataFrame(pd.DataFrame(knn.predict(words_test_fre.values)).values.reshape(-1), columns = ['Predict'])
pred_.index = range(1, len(pred_) + 1)
y_test.index = range(1, len(y_test)+ 1)
y_test = y_test.to_frame(name = 'Actual')
pd.concat([pred_, y_test], axis = 1)