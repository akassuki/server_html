import logging
import torch
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import LSTMModel
from config import LOOK_BACK, LSTM_EPOCHS, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, MIN_TRAIN_POINTS
from utils.helpers import extract_numeric_values_and_keys
from services.db_service import get_connection
import json

logger = logging.getLogger("scheduler")

model_dict = {}
scaler_dict = {}
lock = torch.multiprocessing.Lock()

# Thứ tự cố định các chỉ số (phải khớp với bảng forecasts)
FEATURE_ORDER = ['pm1', 'pm25', 'pm10', 'co', 'co2', 'temperature', 'humidity', 'tvoc', 'no2']

def train_lstm_for_node(node: str):
    """
    Train LSTM cho node dùng dữ liệu gần nhất.
    Trả về dict dự báo đơn giản (không cần ngày).
    """
    global model_dict, scaler_dict
    with lock:
        try:
            conn = get_connection()
            # Lấy 300 bản ghi mới nhất
            df = pd.read_sql_query(
                """
                SELECT raw_payload AS payload, timestamp
                FROM measurements
                WHERE node = ?
                ORDER BY timestamp DESC
                """,
                conn,
                params=(node,)
            )
            conn.close()

            num_points = len(df)
            if num_points < MIN_TRAIN_POINTS:
                logger.warning(f"Không đủ dữ liệu để train LSTM cho {node} (chỉ {num_points}/{MIN_TRAIN_POINTS})")
                return None

            logger.info(f"Train LSTM cho {node}: dùng {num_points} điểm gần nhất")

            # Xử lý LOOK_BACK
            if isinstance(LOOK_BACK, str) and LOOK_BACK == "full_yesterday":
                df_input = df
                lb = len(df) - 1
            elif isinstance(LOOK_BACK, int):
                lb = LOOK_BACK
                df_input = df.head(lb + 1)  # lấy đầu tiên vì đã DESC
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
                logger.warning(f"Không có giá trị hợp lệ nào cho {node}")
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
                logger.warning(f"Không đủ chuỗi để train LSTM cho {node}")
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

            # Dự báo điểm tiếp theo
            last_sequence = scaled_data[:lb].reshape(1, lb, input_size)  # dùng đầu tiên vì DESC
            last_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)

            model.eval()
            with torch.no_grad():
                predicted_scaled = model(last_tensor).cpu().numpy().flatten()

            predicted_real = scaler.inverse_transform([predicted_scaled])[0]

# Clip giá trị về khoảng thực tế (ví dụ min=0 cho nồng độ)
            predicted_real = np.clip(predicted_real, 0, None)  # Không cho âm

# Hoặc clip theo min/max thực tế từ dữ liệu huấn luyện
            min_vals = data_array.min(axis=0)
            max_vals = data_array.max(axis=0)
            predicted_real = np.clip(predicted_real, min_vals, max_vals)

            # Trả về dict đơn giản (không cần ngày)
            forecast_data = {
                "node": node,
                "pm1_avg": round(float(predicted_real[0]), 2),
                "pm25_avg": round(float(predicted_real[1]), 2),
                "pm10_avg": round(float(predicted_real[2]), 2),
                "co_avg": round(float(predicted_real[3]), 2),
                "co2_avg": round(float(predicted_real[4]), 2),
                "temperature_avg": round(float(predicted_real[5]), 2),
                "humidity_avg": round(float(predicted_real[6]), 2),
                "tvoc_avg": round(float(predicted_real[7]), 2),
                "no2_avg": round(float(predicted_real[8]), 2),
            }

            logger.info(f"Dự báo gần nhất cho {node}: {forecast_data}")
            return forecast_data

        except Exception as e:
            logger.error(f"Lỗi train LSTM cho {node}: {str(e)}")
            return None
        