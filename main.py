from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import os
import threading
import schedule
import time
import json
import torch
import pandas as pd
import numpy as np
from config import *
from services.db_service import * 
from services.train_service import *
from services.cluster_service import *
from utils.helpers import *
from flask.json.provider import DefaultJSONProvider
from flask import Flask, request, jsonify, Response

# === TẠO THƯ MỤC LOGS ===
os.makedirs(LOG_DIR, exist_ok=True)

# === XÓA FILE LOG VÀ DB CŨ ===
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
        print(f"Đã xóa file log cũ: {LOG_FILE}")
    except Exception as e:
        print(f"Lỗi xóa log: {e}")

if os.path.exists(DB_PATH):
    try:
        os.remove(DB_PATH)
        print(f"ĐÃ XÓA TOÀN BỘ DATABASE: {DB_PATH}")
    except Exception as e:
        print(f"Lỗi xóa DB: {e}")

# === CONFIG LOGGER NGAY SAU KHI XÓA FILE ===
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

thread_logger = logging.getLogger("scheduler")
thread_logger.setLevel(LOG_LEVEL)

thread_log_file = os.path.join(LOG_DIR, "thread.log")


if os.path.exists(thread_log_file):
    try:
        os.remove(thread_log_file)
    except Exception as e:
        print(f"Lỗi xóa thread.log: {e}")

thread_file_handler = logging.FileHandler(thread_log_file)
thread_file_handler.setLevel(LOG_LEVEL)

thread_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
)
thread_file_handler.setFormatter(thread_formatter)

thread_logger.propagate = False
thread_logger.addHandler(thread_file_handler)



logger.info("=== BẮT ĐẦU CONFIG SERVER ===")

app = Flask(__name__)

CORS(app)

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json = NumpyJSONProvider(app)

init_db()

# def daily_task():
#     logger.info(f"daily_task được kích hoạt lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +07")
#     logger.info("Bắt đầu xử lý hàng ngày: Spectral Clustering + LSTM training + lưu DB...")

#     nodes = get_all_nodes()
#     if not nodes:
#         logger.warning("Không tìm thấy node nào trong DB → bỏ qua daily task")
#         return

#     processed = 0
#     errors = 0

#     for node in nodes:
#         try:
#             logger.info(f"Xử lý node: {node}")

#             #1. Chạy Spectral Clustering và lưu vào cluster_averages
#             cluster_result = run_spectral_clustering_for_node(node)
#             if cluster_result and cluster_result.get("num_clusters", 0) > 0:
#                 insert_cluster_averages(node,cluster_result)
#             else:
#                 logger.info(f"Không có cụm nào cho {node} hôm nay")

#             # 2. Chạy LSTM training và dự báo, lưu vào forecasts
#             forecast_result = train_lstm_for_node(node)
#             if forecast_result:
#                 insert_forecasts(node, forecast_result)  # Truyền node + payload
#             else:
#                 logger.warning(f"Không có dự báo từ LSTM cho {node} → bỏ qua insert forecasts")
#             processed += 1

#         except Exception as e:
#             logger.error(f"Lỗi xử lý node {node} trong daily_task: {str(e)}", exc_info=True)
#             errors += 1

#     logger.info(f"Hoàn tất daily_task: {processed} node xử lý thành công, {errors} node lỗi")

# #schedule.every().day.at(f"{DAILY_TRAIN_HOUR:02d}:{DAILY_TRAIN_MINUTE:02d}").do(daily_task)
# schedule.every(1).minutes.do(daily_task)

# def run_scheduler():
#     while True:
#         schedule.run_pending()
#         time.sleep(60)

# threading.Thread(target=run_scheduler, daemon=True).start()

def daily_task():
    thread_logger.info("=" * 60)
    thread_logger.info(
        f"daily_task START | time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +07"
    )
    thread_logger.info(
        "Pipeline: Spectral Clustering LSTM Training "
    )

    nodes = get_all_nodes()
    if not nodes:
        thread_logger.warning(
            "KHÔNG CÓ NODE TRONG DB -> daily_task KẾT THÚC SỚM"
        )
        thread_logger.info("=" * 60)
        return

    thread_logger.info(
        f"TÌM THẤY {len(nodes)} NODE: {nodes}"
    )

    processed = 0
    errors = 0

    for node in nodes:
        thread_logger.info("-" * 40)
        thread_logger.info(f"NODE START: {node}")

        try:
            # ===== SPECTRAL CLUSTERING =====
            thread_logger.info(
                f"[{node}] Start Spectral Clustering"
            )

            cluster_result = run_spectral_clustering_for_node(node)

            if cluster_result and cluster_result.get("num_clusters", 0) > 0:
                insert_cluster_averages(node, cluster_result)
                thread_logger.info(
                    f"[{node}] CLUSTERING OK | "
                    f"num_clusters={cluster_result.get('num_clusters')}"
                )
            else:
                thread_logger.warning(
                    f"[{node}] CLUSTERING SKIP | "
                    f"không tạo được cụm hợp lệ"
                )

            # ===== LSTM TRAINING =====
            thread_logger.info(
                f"[{node}] Start LSTM Training"
            )

            forecast_result = train_lstm_for_node(node)

            if forecast_result:
                insert_forecasts(node, forecast_result)
                thread_logger.info(
                    f"[{node}] LSTM OK | "
                    f"số bước dự báo={len(forecast_result)}"
                )
            else:
                thread_logger.warning(
                    f"[{node}] LSTM SKIP | không tạo được forecast"
                )

            processed += 1
            thread_logger.info(f"NODE DONE: {node}")

        except Exception as e:
            errors += 1
            thread_logger.error(
                f"LỖI NODE {node}: {str(e)}",
                exc_info=True
            )

    thread_logger.info("-" * 40)
    thread_logger.info(
        f"daily_task FINISHED | "
        f"success={processed} | error={errors}"
    )
    thread_logger.info("=" * 60)




# schedule.every().day.at(f"{DAILY_TRAIN_HOUR:02d}:{DAILY_TRAIN_MINUTE:02d}").do(daily_task)
schedule.every(1).minutes.do(daily_task)


def run_scheduler():
    thread_logger.info("Scheduler thread STARTED")
    while True:
        schedule.run_pending()
        time.sleep(60)


scheduler_thread = threading.Thread(
    target=run_scheduler,
    daemon=True,
    name="SchedulerThread"
)
scheduler_thread.start()



@app.route('/')
def home():
    return "Air Quality Server đang chạy!<br>POST dữ liệu tại /api/data"

@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Dữ liệu JSON không hợp lệ"}), 400

        node = data.pop('node', 'UNKNOWN')
        now = datetime.now()

        # Lưu dữ liệu thô vào bảng measurements
        insert_data(node, data, now)

        #daily_task()

        # Tính giá trị trung bình 
        averages = get_current_average(node)
        if averages:
            insert_daily_averages(node, averages)


        # Log để kiểm tra
        log_parts = [f"{k}: {v}" for k, v in data.items()]
        logger.info(f"[{now.strftime('%d/%m/%Y %H:%M:%S')}] {node} → " + " | ".join(log_parts))

        return jsonify({"status": "OK", "message": "Dữ liệu đã được lưu vào measurements"}), 200

    except Exception as e:
        logger.error(f"Lỗi nhận/lưu dữ liệu từ {node}: {str(e)}")
        return jsonify({"error": "Lỗi server khi lưu dữ liệu"}), 500
    
@app.route("/api/dashboard/realtime")
def dashboard_realtime():
    return jsonify({
        "NODE_1": get_dashboard_node("NODE_01"),
        "NODE_2": get_dashboard_node("NODE_02"),
        "NODE_3": get_dashboard_node("NODE_03"),
    })

@app.route('/api/history', methods=['GET'])
def history_api():
    try:
        # Lấy tham số từ URL (ví dụ: /api/history?node=NODE_01&page=1)
        node = request.args.get('node', 'NODE_01')
        start_date = request.args.get('start_date') # Định dạng YYYY-MM-DD
        end_date = request.args.get('end_date')     # Định dạng YYYY-MM-DD
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 15))
        
        result = get_history_data(node, start_date, end_date, page, limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_data():
    conn = get_connection()
    try:
        node = request.args.get('node', 'NODE_01')
        
        # Query lấy toàn bộ dữ liệu của trạm (Sắp xếp mới nhất trước)
        query = "SELECT timestamp, pm1, pm25, pm10, co, co2, no2, tvoc, temperature, humidity FROM measurements WHERE node = ? ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn, params=(node,))
        
        if df.empty:
            return "Không có dữ liệu để xuất", 404

        # Đổi tên cột cho đẹp (Tiếng Việt)
        df = df.rename(columns={
            "timestamp": "Thời gian",
            "pm1": "PM 1.0 (µg/m³)",
            "pm25": "PM 2.5 (µg/m³)",
            "pm10": "PM 10 (µg/m³)",
            "co": "CO (ppm)",
            "co2": "CO2 (ppm)",
            "no2": "NO2 (ppb)",
            "tvoc": "TVOC (ppb)",
            "temperature": "Nhiệt độ (°C)",
            "humidity": "Độ ẩm (%)"
        })

        # Chuyển thành CSV (encoding='utf-8-sig' để Excel hiển thị đúng tiếng Việt)
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')

        # Trả về file để trình duyệt tải xuống
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=Bao_Cao_{node}.csv"}
        )
    except Exception as e:
        logger.error(f"Lỗi xuất file: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/api/status')
def status():
    return jsonify({
        "status": "running",
        "note": "Dữ liệu sẽ bị xóa hoàn toàn khi server reset hoặc tắt"
    })

if __name__ == '__main__':
    logger.info("=== AIR QUALITY SERVER KHỞI ĐỘNG ===")
    logger.info(f"Truy cập: http://<IP-của-máy>:{PORT}")
    logger.info("LƯU Ý: Khi server tắt hoặc reset, TOÀN BỘ dữ liệu sẽ bị xóa hết")
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)
