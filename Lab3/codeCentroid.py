import pandas as pd
import numpy as np

# tạo hàm lấy dữ liệu
def loadExcel(filename) -> pd.DataFrame:
    '''Hàm này dùng để tải dữ liệu từ một tệp CSV'''
    try:
        # Sử dụng pandas để đọc file CSV và trả về một DataFrame
        data = pd.read_excel(filename)
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
def splitTrainTest(data, target, ratio=0.25):
    from sklearn.model_selection import train_test_split
    data_X = data.drop([target], axis=1)
    data_y = data[[target]]
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    data_train, data_test, y_train, y_test = train_test_split(data_X, data_y, test_size=ratio, random_state=42)
    
    return data_train, data_test, y_train, y_test

def mean_class(data_train, target): # tên cột target, data_train là dạng pandas
    df_group = data_train.groupby(by = target).mean() # tất cả các cột đều dạng số, --> frame # sắp xếp theo bảng chữ cái tăng dần(mặc định)
    return df_group # kết quả là dataframe
# hàm dự đoán dùng khoảng cách euclid
def target_pred(data_group, data_test):
    dict_ = dict()
    
    for index, value in enumerate(data_group.values):
        result = np.sqrt(np.sum((data_test.values - value) ** 2, axis=1))  # khoảng cách euclid
        dict_[index] = result  # Lưu trữ kết quả vào từ điển
    
    # Chuyển từ điển thành DataFrame để dễ thao tác
    df = pd.DataFrame(dict_)  
    
    # Tìm chỉ số của giá trị nhỏ nhất trong mỗi hàng (dòng) để dự đoán lớp
    return df.idxmin(axis=1)

    
##### Có thể phát triển: cho thêm một tham số metric vào hàm, nếu là euclid thì dùng khoảng cách euclid, manhattan thì dùng khoảng cách manhattan.

data = loadExcel('Iris.xls')
data

data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio = 0.3)
print(data_train)
print(X_test)
print(y_test)

df_group = mean_class(data_train, 'iris')
df_group

#  tính khoảng cách và trả về kết quả lớp có khoảng cách gần nhất
df1 = pd.DataFrame(target_pred(df_group, X_test.values), columns = ['Predict'])
df1

# set index y_test để nối 2 frame
y_test.index = range(0, len(y_test))
y_test.columns = ['Actual']
y_test

df2 = pd.DataFrame(y_test)
df2

pd.concat([df1, df2],axis = 1)