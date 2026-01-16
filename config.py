import os
from pathlib import Path

# Đường dẫn cơ sở của dự án
BASE_DIR = Path(__file__).parent

# Đường dẫn database - sẽ bị xóa khi server reset
DB_PATH = str(BASE_DIR / "sensor_data.db")
# DB_PATH = "/mnt/ssd/sensor_data.db"  # Uncomment khi dùng Pi + SSD

# Cấu hình Flask
PORT = 3000

# LSTM - dự báo trung bình ngày mai
LOOK_BACK : str | int= "full_yesterday"        # "full_yesterday" hoặc số cố định 
LSTM_EPOCHS = 5
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 1

# Dự báo trung bình ngày mai
FORECAST_STEPS = 360                # Dự báo 6 giờ trực tiếp
FORECAST_MULTIPLIER = 4.0           # Ước lượng 24 giờ (6h × 4)

# Ngưỡng tối thiểu dữ liệu
MIN_TRAIN_POINTS = 300              # Tối thiểu để train LSTM #@
MIN_PREDICT_POINTS = 60             # Tối thiểu để dự báo

# Spectral Clustering
CLUSTER_MIN_POINTS = 120 #@
CLUSTER_MAX_CLUSTERS = 10
CLUSTER_USE_FULL_DAY = True
CLUSTER_MAX_POINTS = 1440
CLUSTER_DOWNSAMPLE_IF_EXCEED = 2

# Lịch chạy background
DAILY_TRAIN_HOUR = 0
DAILY_TRAIN_MINUTE = 2

# Logging - file log bị xóa khi reset
LOG_DIR = str(BASE_DIR / "logs")
LOG_FILE = os.path.join(LOG_DIR, "server.log")
LOG_LEVEL = "INFO"
