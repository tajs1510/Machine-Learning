import numpy as np
import pandas as pd

def loadCsv(filename):
  try:
    data = pd.read_csv(filename)
    return data
  except FileNotFoundError:
      print(f"Tệp '{filename}' không tồn tại.")
      return pd.DataFrame()
  except pd.errors.EmptyDataError:
      print("Tệp trống hoặc không có dữ liệu hợp lệ.")
      return pd.DataFrame()
  except Exception as e:
      print(f"Đã xảy ra lỗi: {e}")
      return pd.DataFrame()
def transform(data, columns_trans): # data dạng dataframe, data_trans là cột cần biến đổi --> dạng Series, nhiều cột cần biến đổi thì bỏ vào list
    for i in columns_trans:
        unique = data[i].unique() + '-' + i # trả lại mảng
        # tạo ma trận 0
        matrix_0 = np.zeros((len(data), len(unique)), dtype = int)
        frame_0 = pd.DataFrame(matrix_0, columns = unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data[unique] = frame_0
    return data # trả lại data truyền vào nhưng đã bị biến đổi
def scale_data(data, columns_scale): # columns_scale là cột cần scale, nếu nhiều bỏ vào list ['a', 'b']
    for i in columns_scale:
        _max = data[i].max()
        _min = data[i].min()

        min_max_scaller = lambda x: round((x - _min) / (_max - _min), 3)
        data[i] = data[i].apply(min_max_scaller)
    return data # --> trả về frame

def cosine_distance(train_X, test_X): # cả 2 đều dạng mảng
    dict_distance = dict()
    for index, value in enumerate(test_X, start = 1):
        for j in train_X:
            result = np.sqrt(np.sum((j - value)**2))
            if index not in dict_distance:
                dict_distance[index] = [result]
            else:
                dict_distance[index].append(result)
    return dict_distance # {1: [6.0, 5.0], 2: [4.6, 3.1]}

def pred_test(k, train_X, test_X, train_y): # train_X, test_X là mảng, train_y là Series
    lst_predict = list()
    dict_distance = cosine_distance(train_X, test_X)
    train_y = train_y.to_frame(name = 'target').reset_index(drop = True) # train_y là frame
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis = 1)
    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by = i, ascending = True)[:k] # sắp xếp và lấy k
        target_predict = sort_distance['target'].value_counts(ascending = False).index[0]
        lst_predict.append([i, target_predict])
    return lst_predict

data = loadCsv('drug200.csv')
data.head()

df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis = 1)
df

scale_data(df, ['Age', 'Na_to_K']) # không cần gán biến vì nó trả về data đã truyền vào nhưng đã bị scale
df

data_X = df.drop(['Drug'], axis = 1).values
data_y = df['Drug']
print(data_X)
print(data_y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = 0)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
print(len(X_train), len(X_test), len(y_train), len(y_test))
print(type(y_train))

test_pred = pred_test(6, X_train, X_test, y_train)
df_test_pred = pd.DataFrame(test_pred).drop([0], axis = 1)
df_test_pred.index = range(1, len(test_pred) + 1)
df_test_pred.columns = ['Predict']

df_actual = pd.DataFrame(y_test)
df_actual.index = range(1, len(y_test) + 1)
df_actual.columns = ['Actual']

pd.concat([df_test_pred, df_actual], axis = 1)