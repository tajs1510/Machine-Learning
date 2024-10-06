CÂU 1:
 1. Framework : Pandas, Scikit-learn, và CSV
 2. Thuật toán sử dụng:
  - Bernoulli Naive Bayes:
    + Được sử dụng cho các dữ liệu nhị phân (binary data) và đặc biệt hiệu quả khi các tính năng (features) là các biến nhị phân (0 hoặc 1).
    + Thích hợp cho các trường hợp mà việc xuất hiện hoặc không xuất hiện của một từ trong văn bản là quan trọng hơn là tần suất xuất hiện của nó.
  - Multinomial Naive Bayes:
   + Thích hợp cho các bài toán phân loại văn bản, đặc biệt khi các đặc trưng là tần suất xuất hiện của các từ.
   + Làm việc tốt với các dữ liệu mà có nhiều đặc trưng có giá trị lớn (tần suất từ trong văn bản).
 3. Kết quả:
![image](https://github.com/user-attachments/assets/3ea100fc-f7c5-49c6-a5fe-924a57aa2b58)

 4. So sánh 2 kết quả
    Ta thấy được thuật toán Bernoulli Naive Bayes cho ra tỉ lệ là 0.54 trong khi thuật toán Multinomial Naive Bayes là 0.63

CÂU 2:
 1. Framework : Pandas, Scikit-learn
 2. Thuật toán sử dụng:
  - Bernoulli Naive Bayes và Multinomial Naive Bayes:
    + Được sử dụng cho bài toán phân loại văn bản.
    + Bernoulli thích hợp cho dữ liệu nhị phân (có hoặc không có từ), trong khi Multinomial thường dùng cho dữ liệu với tần suất từ.
  - Gaussian Naive Bayes:
   + Thích hợp cho dữ liệu mà các tính năng có phân phối chuẩn (Gaussian).
   + Sử dụng cho dữ liệu số và có thể hoạt động tốt hơn khi có các đặc trưng liên tục, như tuổi tác hoặc nồng độ.
