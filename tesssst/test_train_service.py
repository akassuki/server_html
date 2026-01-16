# test_train_service_new.py
# Chương trình test hàm train_lstm_for_node phiên bản mới (trả về dự báo cho forecasts)

import os
import sqlite3
import json
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Giả lập config (bạn có thể thay đổi)
LOOK_BACK = "full_yesterday"  # hoặc số, ví dụ 60
LSTM_EPOCHS = 10              # Tăng lên để train ổn định hơn
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 1
MIN_TRAIN_POINTS = 5

# Giả lập logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Biến toàn cục giống file gốc
model_dict = {}
scaler_dict = {}
lock = torch.multiprocessing.Lock()

# Thứ tự cố định chỉ số (phải khớp với bảng forecasts)
FEATURE_ORDER = ['pm1', 'pm25', 'pm10', 'co', 'co2', 'temperature', 'humidity', 'tvoc', 'no2']

# Class LSTMModel (copy từ models/lstm_model.py)
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Hàm train_lstm_for_node phiên bản mới (đã sửa để trả về dự báo forecasts)
def train_lstm_for_node(node: str):
    """
    Train LSTM cho node và dự báo giá trị trung bình ngày hôm sau.
    Trả về dict khớp với bảng forecasts để dễ insert.
    """
    global model_dict, scaler_dict

    with lock:
        try:
            conn = sqlite3.connect("test_sensor_data.db")
            df = pd.read_sql_query(
                "SELECT payload, timestamp FROM measurements WHERE node=? ORDER BY timestamp",
                conn, params=(node,)
            )
            conn.close()

            yesterday_start = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            yesterday_end = yesterday_start + timedelta(days=1) - timedelta(seconds=1)

            df_yesterday = df[
                (pd.to_datetime(df['timestamp']) >= yesterday_start) &
                (pd.to_datetime(df['timestamp']) < yesterday_end)
            ]

            num_points = len(df_yesterday)
            if num_points < MIN_TRAIN_POINTS:
                logger.warning(f"Không đủ dữ liệu để train LSTM cho {node} (chỉ {num_points}/{MIN_TRAIN_POINTS})")
                return None

            logger.info(f"Train LSTM cho {node}: dùng {num_points} điểm ngày hôm trước")

            # Xử lý LOOK_BACK
            if isinstance(LOOK_BACK, str) and LOOK_BACK == "full_yesterday":
                df_input = df_yesterday
                lb = len(df_yesterday) - 1
            elif isinstance(LOOK_BACK, int):
                lb = LOOK_BACK
                df_input = df.tail(lb + 1)
            else:
                logger.error(f"LOOK_BACK không hợp lệ: {LOOK_BACK}")
                return None

            all_values = []
            for payload_str in df_input['payload']:
                payload = json.loads(payload_str)
                values = [
                    payload.get('pm1') or payload.get('pm1.0') or 0.0,
                    payload.get('pm25') or 0.0,
                    payload.get('pm10') or 0.0,
                    payload.get('co') or 0.0,
                    payload.get('co2') or 0.0,
                    payload.get('temperature') or 0.0,
                    payload.get('humidity') or 0.0,
                    payload.get('tvoc') or payload.get('TVOC') or 0.0,
                    payload.get('no2') or 0.0
                ]
                all_values.append(values)

            if not all_values:
                return None

            data_array = np.array(all_values)
            scaler = MinMaxScaler()
            scaler.fit(data_array)
            scaler_dict[node] = scaler

            scaled_data = scaler.transform(data_array)

            X, y = [], []
            for i in range(len(scaled_data) - lb):
                X.append(scaled_data[i:i + lb])
                y.append(scaled_data[i + lb])

            if len(X) == 0:
                return None

            X = np.array(X)
            y = np.array(y)

            input_size = X.shape[2]
            model = LSTMModel(input_size, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

            for epoch in range(LSTM_EPOCHS):
                model.train()
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

            model_dict[node] = model
            logger.info(f"Train LSTM thành công cho {node} | Loss cuối: {loss.item():.6f}")

            # Dự báo điểm tiếp theo (đại diện ngày mai)
            last_sequence = scaled_data[-lb:].reshape(1, lb, input_size)
            last_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)

            model.eval()
            with torch.no_grad():
                predicted_scaled = model(last_tensor).cpu().numpy().flatten()

            predicted_real = scaler.inverse_transform([predicted_scaled])[0]

            tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            forecast_data = {
                "node": node,
                "forecast_date": tomorrow_date,
                "pm1_avg": round(float(predicted_real[0]), 2),
                "pm25_avg": round(float(predicted_real[1]), 2),
                "pm10_avg": round(float(predicted_real[2]), 2),
                "co_avg": round(float(predicted_real[3]), 2),
                "co2_avg": round(float(predicted_real[4]), 2),
                "temperature_avg": round(float(predicted_real[5]), 2),
                "humidity_avg": round(float(predicted_real[6]), 2),
                "tvoc_avg": round(float(predicted_real[7]), 2),
                "no2_avg": round(float(predicted_real[8]), 2),
                "calculated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(f"Dự báo ngày mai cho {node} (forecast_date={tomorrow_date}): {forecast_data}")

            return forecast_data

        except Exception as e:
            logger.error(f"Lỗi train LSTM cho {node}: {str(e)}")
            return None

# PHẦN TẠO DỮ LIỆU GIẢ LẬP
def create_test_database():
    db_name = "test_sensor_data.db"
    if os.path.exists(db_name):
        os.remove(db_name)

    conn = sqlite3.connect(db_name)
    conn.execute('''
        CREATE TABLE measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            payload TEXT
        )
    ''')

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    test_data = []

    # Tạo 10 bản ghi cho NODE_01
    for i in range(10):
        hour = 8 + i
        payload = {
            "pm1": 4.0 + i*0.5,
            "pm25": 10.0 + i*1.0,
            "pm10": 18.0 + i*1.5,
            "co": 0.3 + i*0.05,
            "co2": 380 + i*10,
            "temperature": 24.0 + i*0.5,
            "humidity": 55.0 + i*2.0,
            "tvoc": 90 + i*5,
            "no2": 8 + i*1.0
        }
        timestamp = f"{yesterday} {hour:02d}:00:00"
        test_data.append(("NODE_01", timestamp, json.dumps(payload)))

    # Tạo 10 bản ghi cho NODE_02
    for i in range(10):
        hour = 8 + i
        payload = {
            "pm1": 3.5 + i*0.4,
            "pm25": 9.0 + i*0.8,
            "pm10": 16.0 + i*1.2,
            "co": 0.25 + i*0.04,
            "co2": 360 + i*8,
            "temperature": 23.5 + i*0.4,
            "humidity": 58.0 + i*1.5,
            "tvoc": 85 + i*4,
            "no2": 7 + i*0.8
        }
        timestamp = f"{yesterday} {hour:02d}:00:00"
        test_data.append(("NODE_02", timestamp, json.dumps(payload)))

    # Tạo 10 bản ghi cho NODE_03
    for i in range(10):
        hour = 8 + i
        payload = {
            "pm1": 5.0 + i*0.6,
            "pm25": 13.0 + i*1.2,
            "pm10": 22.0 + i*1.8,
            "co": 0.35 + i*0.06,
            "co2": 400 + i*12,
            "temperature": 25.5 + i*0.6,
            "humidity": 52.0 + i*2.5,
            "tvoc": 95 + i*6,
            "no2": 9 + i*1.2
        }
        timestamp = f"{yesterday} {hour:02d}:00:00"
        test_data.append(("NODE_03", timestamp, json.dumps(payload)))

    conn.executemany("INSERT INTO measurements (node, timestamp, payload) VALUES (?, ?, ?)", test_data)
    conn.commit()
    conn.close()
    logger.info("Đã tạo database test với 30 bản ghi (10 bản ghi mỗi node) cho ngày hôm trước")

# Chạy test
if __name__ == "__main__":
    create_test_database()

    nodes_to_test = ["NODE_01", "NODE_02", "NODE_03"]

    for node in nodes_to_test:
        logger.info(f"\n=== BẮT ĐẦU TRAIN NODE: {node} ===")
        forecast_result = train_lstm_for_node(node)

        if forecast_result:
            logger.info(f"Train thành công và có dự báo cho {node}")
            logger.info(f"Forecast date: {forecast_result['forecast_date']}")
            logger.info(f"pm1_avg: {forecast_result['pm1_avg']}, pm25_avg: {forecast_result['pm25_avg']}, ...")
            logger.info(f"Full forecast dict: {forecast_result}")
        else:
            logger.info(f"Không train được hoặc không có dự báo cho {node}")

        logger.info(f"=== KẾT THÚC TRAIN NODE: {node} ===\n")

    # Kiểm tra model_dict & scaler_dict
    print("\nKết quả cuối cùng:")
    print(f"Số node đã train thành công: {len(model_dict)}")
    for node in model_dict:
        print(f"- Node {node}: model tồn tại = {model_dict[node] is not None}, scaler tồn tại = {scaler_dict.get(node) is not None}")

    print("\nTest hoàn tất. Kiểm tra log để xem chi tiết loss và dự báo.")
    