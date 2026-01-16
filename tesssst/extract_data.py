import sqlite3
import pandas as pd
import os

# ===============================
# Kết nối Database
# ===============================
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'sensor_data.db')
)
print(f"Đang kết nối DB tại: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

# ===============================
# 1. Measurements
# ===============================
print("\n=== Measurements (10 bản ghi mới nhất) ===")
df_meas = pd.read_sql_query(
    "SELECT * FROM measurements ORDER BY timestamp DESC LIMIT 10",
    conn
)
print(df_meas)

# ===============================
# 2. Forecasts
# ===============================
print("\n=== Forecasts (5 bản ghi mới nhất) ===")
df_fc = pd.read_sql_query(
    "SELECT * FROM forecasts ORDER BY id DESC LIMIT 5",
    conn
)
print(df_fc)

# ===============================
# 3. Cluster Averages
# ===============================
print("\n=== Cluster Averages (kết quả clustering mới nhất) ===")

df_cl = pd.read_sql_query(
    "SELECT * FROM cluster_averages ORDER BY id DESC",
    conn
)
print(df_cl)

# ===============================
# 4. Daily Averages
# ===============================
print("\n=== Daily Averages (trung bình tích lũy) ===")
df_daily = pd.read_sql_query(
    "SELECT * FROM daily_averages",
    conn
)
print(df_daily)

conn.close()
print("\n✅ Hoàn tất trích xuất dữ liệu.")
