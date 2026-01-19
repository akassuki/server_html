import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_size, hidden_size=50, num_layers=2):
    """
    Tạo mô hình LSTM bằng Keras/TensorFlow.
    - input_size: số chỉ số đầu vào (ví dụ 9: pm1, pm25, ..., no2)
    - hidden_size: số unit ẩn mỗi lớp LSTM (mặc định 50)
    - num_layers: số lớp LSTM xếp chồng (mặc định 2)
    """
    model = Sequential()

    # Lớp LSTM đầu tiên (phải chỉ định input_shape)
    model.add(LSTM(
        units=hidden_size,
        return_sequences=(num_layers > 1),  # chỉ return sequences nếu còn lớp sau
        input_shape=(None, input_size)      # None = sequence length linh hoạt
    ))

    # Thêm các lớp LSTM tiếp theo nếu num_layers > 1
    for _ in range(1, num_layers):
        model.add(LSTM(units=hidden_size, return_sequences=False))

    # Lớp Dense cuối để dự báo output cùng số chiều input
    model.add(Dense(input_size))

    # Compile model (dùng MSE loss cho dự báo số)
    model.compile(optimizer='adam', loss='mse')

    return model