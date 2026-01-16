import sqlite3
import pandas as pd
import numpy as np
import json
from config import DB_PATH
import logging
import datetime

logger = logging.getLogger(__name__)

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    
    # Tạo bảng measurements - ĐẦY ĐỦ 9 CHỈ SỐ
    conn.execute('''
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        pm1 REAL,
        pm25 REAL,
        pm10 REAL,
        co REAL,
        co2 REAL,
        temperature REAL,
        humidity REAL,
        tvoc REAL,
        no2 REAL,
        raw_payload TEXT
    )
    ''')

    # Tạo bảng forecasts
    conn.execute('''
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node TEXT NOT NULL,
        pm1_avg REAL,
        pm25_avg REAL,
        pm10_avg REAL,
        co_avg REAL,
        co2_avg REAL,
        temperature_avg REAL,
        humidity_avg REAL,
        tvoc_avg REAL,
        no2_avg REAL
    )
    ''')

    # Tạo bảng cluster_averages
    conn.execute('''
    CREATE TABLE IF NOT EXISTS cluster_averages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node TEXT NOT NULL,
        cluster_num INTEGER NOT NULL,
        cluster_id INTERGER NOT NULL,
        point_count INTEGER,
        silhouette_score FLOAT,
        pm1_avg REAL,
        pm25_avg REAL,
        pm10_avg REAL,
        co_avg REAL,
        co2_avg REAL,
        temperature_avg REAL,
        humidity_avg REAL,
        tvoc_avg REAL,
        no2_avg REAL,
        UNIQUE(node, cluster_id)
    )
    ''')

    # Tạo bảng daily_averages
    conn.execute('''
    CREATE TABLE IF NOT EXISTS daily_averages (
        node TEXT PRIMARY KEY,
        pm1_avg REAL,
        pm25_avg REAL,
        pm10_avg REAL,
        co_avg REAL,
        co2_avg REAL,
        temperature_avg REAL,
        humidity_avg REAL,
        tvoc_avg REAL,
        no2_avg REAL
    )
    ''')

    # Tạo INDEX riêng biệt
    conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_node_timestamp ON measurements (node, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements (timestamp)")


    conn.commit()
    conn.close()

def insert_data(node: str, payload: dict, timestamp):
    conn = get_connection()
    try:
        # Lấy giá trị theo tên key cố định (không phụ thuộc thứ tự)
        pm1 = payload.get('pm1') or payload.get('pm1.0') or payload.get('PM1') or payload.get('pm1_0') or None
        pm25 = payload.get('pm25') or payload.get('PM2.5') or payload.get('pm2_5') or None
        pm10 = payload.get('pm10') or payload.get('PM10') or None
        co = payload.get('co') or payload.get('CO') or None
        co2 = payload.get('co2') or payload.get('CO2') or None
        temperature = payload.get('temperature') or payload.get('temp') or payload.get('nhiet_do') or None
        humidity = payload.get('humidity') or payload.get('do_am') or payload.get('humi') or None
        tvoc = payload.get('tvoc') or payload.get('TVOC') or None
        no2 = payload.get('no2') or payload.get('NO2') or None

        raw_payload = json.dumps(payload)

        conn.execute('''
        INSERT INTO measurements 
        (node, timestamp, pm1, pm25, pm10, co, co2, temperature, humidity, tvoc, no2, raw_payload)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (node, timestamp, pm1, pm25, pm10, co, co2, temperature, humidity, tvoc, no2, raw_payload))
        
        conn.commit()
        logger.info(f"Đã lưu dữ liệu cho node {node} tại {timestamp}")
    except Exception as e:
        logger.error(f"Lỗi insert data cho node {node}: {e}")
    finally:
        conn.close()

def insert_cluster_averages(node: str, payload: dict = None):
    if payload is None or not isinstance(payload, dict):
        logger.warning(f"Không có dữ liệu cụm cho node {node} → bỏ qua")
        return

    conn = get_connection()
    try:
        cur = conn.cursor()
        clusters = payload.get("cluster_averages", [])
        silhouette = payload.get("silhouette_score")
        num_cluster = payload.get("num_clusters")


        if not clusters:
            logger.warning(f"Không có cụm nào để lưu cho node {node}")
            return

        for cl in clusters:
            avgs = cl.get("averages", {})
            cur.execute('''
                INSERT OR REPLACE INTO cluster_averages
                (node, cluster_num, silhouette_score, cluster_id, point_count,
                 pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg,
                 temperature_avg, humidity_avg, tvoc_avg, no2_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node,
                num_cluster,
                silhouette,
                cl["cluster_id"],
                cl["point_count"],
                avgs.get("pm1_0"),
                avgs.get("pm2_5"),
                avgs.get("pm10"),
                avgs.get("co"),
                avgs.get("co2"),
                avgs.get("temp"),
                avgs.get("humi"),
                avgs.get("tvoc"),
                avgs.get("no2")
            ))

        conn.commit()
        logger.info(f"Đã lưu {len(clusters)} cụm cho node {node} (silhouette: {silhouette})")
    except Exception as e:
        logger.error(f"Lỗi insert_cluster_averages cho node {node}: {e}")
    finally:
        conn.close()

def insert_forecasts(node: str,payload: dict = None):
    """
    Lưu hoặc cập nhật giá trị dự đoán cho node.
    payload phải có các key: pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg, temperature_avg,
    humidity_avg, tvoc_avg, no2_avg
    """
    conn = get_connection()
    try:
        pm1_avg = payload.get('pm1_avg')
        pm25_avg = payload.get('pm25_avg')
        pm10_avg = payload.get('pm10_avg')
        co_avg = payload.get('co_avg')
        co2_avg = payload.get('co2_avg')
        temperature_avg = payload.get('temperature_avg')
        humidity_avg = payload.get('humidity_avg')
        tvoc_avg = payload.get('tvoc_avg')
        no2_avg = payload.get('no2_avg')

        conn.execute('''
            INSERT OR REPLACE INTO forecasts
            (node, pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg, temperature_avg, humidity_avg, tvoc_avg, no2_avg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (node, pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg, temperature_avg, humidity_avg, tvoc_avg, no2_avg))
        
        conn.commit()
        logger.info(f"Đã cập nhật giá trị trung bình tích lũy cho node {node}")
    except Exception as e:
        logger.error(f"Lỗi insert_forecasts cho node {node}: {e}")
    finally:
        conn.close()


def insert_daily_averages(node: str, payload: dict):
    """
    Lưu hoặc cập nhật giá trị trung bình tích lũy cho node.
    payload phải có các key: pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg, temperature_avg,
    humidity_avg, tvoc_avg, no2_avg
    """
    conn = get_connection()
    try:
        pm1_avg = payload.get('pm1_avg')
        pm25_avg = payload.get('pm25_avg')
        pm10_avg = payload.get('pm10_avg')
        co_avg = payload.get('co_avg')
        co2_avg = payload.get('co2_avg')
        temperature_avg = payload.get('temperature_avg')
        humidity_avg = payload.get('humidity_avg')
        tvoc_avg = payload.get('tvoc_avg')
        no2_avg = payload.get('no2_avg')

        conn.execute('''
            INSERT OR REPLACE INTO daily_averages
            (node, pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg, temperature_avg, humidity_avg, tvoc_avg, no2_avg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (node, pm1_avg, pm25_avg, pm10_avg, co_avg, co2_avg, temperature_avg, humidity_avg, tvoc_avg, no2_avg))
        
        conn.commit()
        logger.info(f"Đã cập nhật giá trị trung bình tích lũy cho node {node}")
    except Exception as e:
        logger.error(f"Lỗi insert_or_update_daily_average cho node {node}: {e}")
    finally:
        conn.close()


def get_recent_data(node: str, limit: int):
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM measurements WHERE node = ? ORDER BY timestamp DESC LIMIT ?",
            conn, params=(node, limit)
        )
        return df
    finally:
        conn.close()

def get_all_nodes():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT DISTINCT node FROM measurements", conn)
        return df['node'].tolist()
    finally:
        conn.close()

def get_latest(node: str):
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM measurements WHERE node = ? ORDER BY timestamp DESC LIMIT 1",
            conn, params=(node,)
        )
        return df.iloc[0].to_dict() if not df.empty else None
    finally:
        conn.close()

def get_current_average(node: str):
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT AVG(pm1) as pm1_avg, AVG(pm25) as pm25_avg, AVG(pm10) as pm10_avg, 
                      AVG(co) as co_avg, AVG(co2) as co2_avg, AVG(temperature) as temperature_avg,
                      AVG(humidity) as humidity_avg, 
                      AVG(tvoc) as tvoc_avg, AVG(no2) as no2_avg
               FROM measurements WHERE node = ?""",
            conn, params=(node,)
        )
        if df.empty:
            return None

        avg_dict = df.iloc[0].to_dict()
        # Làm tròn tất cả các giá trị float đến 2 chữ số
        rounded_dict = {k: round(v, 2) if v is not None else None for k, v in avg_dict.items()}
        return rounded_dict

    finally:
        conn.close()


def get_dashboard_node(node: str):
    conn = get_connection()
    try:
        # 1. Latest measurement
        latest = pd.read_sql_query(
            """SELECT * FROM measurements
               WHERE node = ?
               ORDER BY timestamp DESC LIMIT 1""",
            conn, params=(node,)
        )

        # 2. History (for chart)
        history = pd.read_sql_query(
            """SELECT timestamp, pm1, pm25, pm10, co, co2,
                      temperature, humidity, tvoc, no2
               FROM measurements
               WHERE node = ?
               ORDER BY timestamp DESC LIMIT 20""",
            conn, params=(node,)
        )

        # 3. Average (daily_averages)
        avg = pd.read_sql_query(
            """SELECT pm1_avg, pm25_avg, pm10_avg,
                      co_avg, co2_avg, temperature_avg,
                      humidity_avg, tvoc_avg, no2_avg
               FROM daily_averages
               WHERE node = ?""",
            conn, params=(node,)
        )

        # === KHẮC PHỤC LỖI JSON NaN TẠI ĐÂY ===
        # Thay thế NaN của Pandas thành None (Python) -> null (JSON)
        if not latest.empty:
            latest = latest.replace({np.nan: None})
        
        if not avg.empty:
            avg = avg.replace({np.nan: None})
            
        if not history.empty:
            history = history.replace({np.nan: None})
        # =======================================

        return {
            "status": "online" if not latest.empty else "offline",
            "timestamp": latest.iloc[0]["timestamp"] if not latest.empty else None,
            # drop các cột không cần thiết trước khi to_dict
            "data": latest.iloc[0].drop(["id", "node", "timestamp"]).to_dict()
                    if not latest.empty else {},
            "avg": avg.iloc[0].to_dict() if not avg.empty else {},
            "history": history.iloc[::-1].to_dict(orient="records")
        }
    finally:
        conn.close()


def get_history_data(node, start_date=None, end_date=None, page=1, limit=15):
    conn = get_connection()
    try:
        offset = (page - 1) * limit
        
        # Xây dựng câu query cơ bản
        query = "SELECT * FROM measurements WHERE node = ?"
        params = [node]
        
        # Thêm điều kiện thời gian nếu có
        if start_date:
            query += " AND date(timestamp) >= date(?)"
            params.append(start_date)
        if end_date:
            query += " AND date(timestamp) <= date(?)"
            params.append(end_date)
            
        # Đếm tổng số bản ghi (để làm phân trang)
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total_records = pd.read_sql_query(count_query, conn, params=params).iloc[0, 0]
        
        # Query lấy dữ liệu chính thức
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Xử lý NaN thành None để JSON không bị lỗi
        df = df.replace({np.nan: None})
        
        return {
            "data": df.to_dict(orient="records"),
            "pagination": {
                "total": int(total_records),
                "page": page,
                "limit": limit,
                "total_pages": (total_records + limit - 1) // limit
            }
        }
    except Exception as e:
        logger.error(f"Lỗi lấy lịch sử: {e}")
        return {"data": [], "pagination": {"total": 0, "page": 1, "total_pages": 0}}
    finally:
        conn.close()



def get_all_forecasts():
    conn = get_connection()
    try:
        # Lấy dữ liệu từ bảng forecasts
        df = pd.read_sql_query("SELECT * FROM forecasts", conn)
        
        # Xử lý NaN thành None để tránh lỗi JSON
        if not df.empty:
            df = df.replace({np.nan: None})
            return df.to_dict(orient="records")
        return []
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu forecast: {e}")
        return []
    finally:
        conn.close()

def get_latest_forecasts():
    conn = get_connection()
    try:
        # Câu query này chỉ lấy bản ghi có ID lớn nhất (mới nhất) cho từng node
        query = """
        SELECT * FROM forecasts 
        WHERE id IN (
            SELECT MAX(id) 
            FROM forecasts 
            GROUP BY node
        )
        """
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            df = df.replace({np.nan: None})
            return df.to_dict(orient="records")
        return []
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu forecast mới nhất: {e}")
        return []
    finally:
        conn.close()

def get_all_clusters_sorted():
    conn = get_connection()
    try:
        # Lấy tất cả cụm, sắp xếp theo Node -> Số lượng điểm giảm dần (DESC)
        query = "SELECT * FROM cluster_averages ORDER BY node, point_count DESC"
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            df = df.replace({np.nan: None})
            return df.to_dict(orient="records")
        return []
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu cluster: {e}")
        return []
    finally:
        conn.close()