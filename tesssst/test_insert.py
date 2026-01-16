# test_insert_forecast_cluster.py
# Chương trình test độc lập + truy vấn toàn bộ dữ liệu trong bảng forecasts và cluster_averages

import sqlite3
import logging
import json
import os
import pandas as pd
from datetime import datetime

# Giả lập logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Database test tạm thời
TEST_DB = "test_forecast_cluster.db"

# Hàm insert_forecasts (copy nguyên từ bạn)
def insert_forecasts(forecast_data: dict, node: str = None):
    if not forecast_data:
        logger.warning("Không có dữ liệu dự báo để insert vào forecasts")
        return False, "Không có dữ liệu dự báo"

    node_to_use = forecast_data.get("node") or node
    if not node_to_use:
        logger.error("Không tìm thấy node trong forecast_data hoặc param")
        return False, "Thiếu thông tin node"

    required_keys = ["node", "forecast_date"]
    if not all(k in forecast_data for k in required_keys):
        logger.error(f"Dữ liệu dự báo thiếu trường bắt buộc cho node {node_to_use}")
        return False, "Dữ liệu dự báo không đầy đủ"

    conn = None
    try:
        conn = sqlite3.connect(TEST_DB)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO forecasts (
                node, forecast_date,
                pm1_avg, pm25_avg, pm10_avg,
                co_avg, co2_avg,
                temperature_avg, humidity_avg,
                tvoc_avg, no2_avg,
                calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            forecast_data["node"],
            forecast_data["forecast_date"],
            forecast_data.get("pm1_avg"),
            forecast_data.get("pm25_avg"),
            forecast_data.get("pm10_avg"),
            forecast_data.get("co_avg"),
            forecast_data.get("co2_avg"),
            forecast_data.get("temperature_avg"),
            forecast_data.get("humidity_avg"),
            forecast_data.get("tvoc_avg"),
            forecast_data.get("no2_avg"),
            forecast_data.get("calculated_at") or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        affected = cur.rowcount
        if affected > 0:
            msg = f"Đã lưu dự báo thành công cho {node_to_use} ngày {forecast_data['forecast_date']}"
            logger.info(msg)
            return True, msg
        else:
            msg = f"Không có thay đổi nào khi insert forecast cho {node_to_use}"
            logger.warning(msg)
            return False, msg
    except Exception as e:
        if conn:
            conn.rollback()
        error_msg = f"Lỗi khi insert forecast cho {node_to_use}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg
    finally:
        if conn:
            conn.close()

# Hàm insert_cluster_averages (copy nguyên từ bạn)
def insert_cluster_averages(result: dict):
    if not result or result.get("num_clusters", 0) == 0:
        logger.info("Không có cụm nào để lưu vào database")
        return False, 0

    clusters = result.get("cluster_averages", [])
    if not clusters:
        logger.warning("Kết quả phân cụm rỗng → không lưu")
        return False, 0

    conn = None
    try:
        conn = sqlite3.connect(TEST_DB)
        cur = conn.cursor()
        inserted_count = 0
        for cluster in clusters:
            required_keys = ["node", "cluster_date", "cluster_id", "point_count"]
            if not all(k in cluster for k in required_keys):
                logger.warning(f"Bỏ qua cụm thiếu thông tin bắt buộc: cluster_id={cluster.get('cluster_id')}")
                continue

            cur.execute('''
                INSERT OR REPLACE INTO cluster_averages (
                    node, cluster_date, cluster_id, point_count,
                    pm1_avg, pm25_avg, pm10_avg,
                    co_avg, co2_avg,
                    temperature_avg, humidity_avg, pressure_avg,
                    tvoc_avg, no2_avg,
                    calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                cluster["node"],
                cluster["cluster_date"],
                cluster["cluster_id"],
                cluster["point_count"],
                cluster.get("pm1_avg"),
                cluster.get("pm25_avg"),
                cluster.get("pm10_avg"),
                cluster.get("co_avg"),
                cluster.get("co2_avg"),
                cluster.get("temperature_avg"),
                cluster.get("humidity_avg"),
                cluster.get("pressure_avg"),
                cluster.get("tvoc_avg"),
                cluster.get("no2_avg")
            ))
            inserted_count += cur.rowcount

        conn.commit()
        first_node = clusters[0].get("node", "unknown") if clusters else "unknown"
        cluster_date = result.get("cluster_date", "unknown")
        num_clusters = result.get("num_clusters", 0)
        logger.info(
            f"Đã lưu thành công {inserted_count} cụm "
            f"cho node {first_node} ngày {cluster_date} "
            f"(tổng {num_clusters} cụm)"
        )
        return True, inserted_count
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Lỗi lưu cluster_averages: {str(e)}", exc_info=True)
        return False, 0
    finally:
        if conn:
            conn.close()

# Tạo database test và bảng
def create_test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
        logger.info(f"Đã xóa DB test cũ: {TEST_DB}")

    conn = sqlite3.connect(TEST_DB)
    # Bảng forecasts
    conn.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node TEXT NOT NULL,
            forecast_date DATE NOT NULL,
            pm1_avg REAL,
            pm25_avg REAL,
            pm10_avg REAL,
            co_avg REAL,
            co2_avg REAL,
            temperature_avg REAL,
            humidity_avg REAL,
            tvoc_avg REAL,
            no2_avg REAL,
            calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node, forecast_date)
        )
    ''')
    # Bảng cluster_averages
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cluster_averages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node TEXT NOT NULL,
            cluster_date DATE NOT NULL,
            cluster_id INTEGER NOT NULL,
            point_count INTEGER NOT NULL,
            pm1_avg REAL,
            pm25_avg REAL,
            pm10_avg REAL,
            co_avg REAL,
            co2_avg REAL,
            temperature_avg REAL,
            humidity_avg REAL,
            pressure_avg REAL,
            tvoc_avg REAL,
            no2_avg REAL,
            calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node, cluster_date, cluster_id)
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Đã tạo database test và các bảng forecasts + cluster_averages")

# Dữ liệu giả lập cho test
def get_test_forecast_data():
    return {
        "node": "NODE_TEST_01",
        "forecast_date": "2026-01-16",
        "pm1_avg": 3.45,
        "pm25_avg": 8.12,
        "pm10_avg": 15.67,
        "co_avg": 0.28,
        "co2_avg": 420.5,
        "temperature_avg": 24.8,
        "humidity_avg": 62.3,
        "tvoc_avg": 85.0,
        "no2_avg": 7.9,
        "calculated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_test_cluster_result():
    return {
        "num_clusters": 3,
        "silhouette_score": 0.6123,
        "total_points": 288,
        "cluster_date": "2026-01-15",
        "cluster_averages": [
            {
                "node": "NODE_TEST_01",
                "cluster_date": "2026-01-15",
                "cluster_id": 0,
                "point_count": 120,
                "pm1_avg": 2.1,
                "pm25_avg": 5.4,
                "pm10_avg": 10.8,
                "co_avg": 0.15,
                "co2_avg": 380.2,
                "temperature_avg": 22.5,
                "humidity_avg": 58.0,
                "pressure_avg": None,
                "tvoc_avg": 60.0,
                "no2_avg": 6.2
            },
            {
                "node": "NODE_TEST_01",
                "cluster_date": "2026-01-15",
                "cluster_id": 1,
                "point_count": 100,
                "pm1_avg": 4.8,
                "pm25_avg": 12.3,
                "pm10_avg": 22.1,
                "co_avg": 0.45,
                "co2_avg": 520.7,
                "temperature_avg": 26.8,
                "humidity_avg": 68.5,
                "pressure_avg": None,
                "tvoc_avg": 120.0,
                "no2_avg": 15.4
            },
            {
                "node": "NODE_TEST_01",
                "cluster_date": "2026-01-15",
                "cluster_id": 2,
                "point_count": 68,
                "pm1_avg": 7.2,
                "pm25_avg": 18.9,
                "pm10_avg": 35.6,
                "co_avg": 0.72,
                "co2_avg": 680.4,
                "temperature_avg": 29.1,
                "humidity_avg": 75.2,
                "pressure_avg": None,
                "tvoc_avg": 210.5,
                "no2_avg": 22.8
            }
        ]
    }

# Truy vấn và in toàn bộ dữ liệu trong bảng
def query_all_data():
    conn = sqlite3.connect(TEST_DB)
    try:
        # 1. Schema bảng forecasts
        print("\n=== SCHEMA BẢNG forecasts ===")
        schema_fc = pd.read_sql_query("PRAGMA table_info(forecasts)", conn)
        print(schema_fc.to_string(index=False))

        # 2. Toàn bộ dữ liệu forecasts
        print("\n=== TOÀN BỘ DỮ LIỆU TRONG BẢNG forecasts ===")
        df_fc = pd.read_sql_query("SELECT * FROM forecasts ORDER BY calculated_at DESC", conn)
        if df_fc.empty:
            print("→ Bảng forecasts hiện rỗng")
        else:
            print(df_fc.to_string(index=False))
            print(f"Tổng số bản ghi: {len(df_fc)}")

        # 3. Schema bảng cluster_averages
        print("\n=== SCHEMA BẢNG cluster_averages ===")
        schema_cl = pd.read_sql_query("PRAGMA table_info(cluster_averages)", conn)
        print(schema_cl.to_string(index=False))

        # 4. Toàn bộ dữ liệu cluster_averages
        print("\n=== TOÀN BỘ DỮ LIỆU TRONG BẢNG cluster_averages ===")
        df_cl = pd.read_sql_query("SELECT * FROM cluster_averages ORDER BY calculated_at DESC", conn)
        if df_cl.empty:
            print("→ Bảng cluster_averages hiện rỗng")
        else:
            print(df_cl.to_string(index=False))
            print(f"Tổng số bản ghi: {len(df_cl)}")

    except Exception as e:
        print(f"Lỗi khi truy vấn DB: {e}")
    finally:
        conn.close()

# Chạy test
if __name__ == "__main__":
    logger.info("=== BẮT ĐẦU TEST 2 HÀM INSERT VÀ TRUY VẤN ===")
    create_test_db()

    # Test insert_forecasts
    logger.info("\nTEST 1: insert_forecasts")
    forecast_data = get_test_forecast_data()
    success, msg = insert_forecasts(forecast_data)
    logger.info(f"Kết quả insert forecast: {'Thành công' if success else 'Thất bại'} - {msg}")

    # Test insert_cluster_averages
    logger.info("\nTEST 2: insert_cluster_averages")
    cluster_result = get_test_cluster_result()
    success, count = insert_cluster_averages(cluster_result)
    logger.info(f"Kết quả insert cluster: {'Thành công' if success else 'Thất bại'} - Insert {count} cụm")

    # Truy vấn và in toàn bộ dữ liệu
    logger.info("\nTRUY VẤN TOÀN BỘ DỮ LIỆU SAU INSERT")
    query_all_data()

    logger.info("\n=== TEST HOÀN TẤT ===")
    logger.info("Nếu bảng có dữ liệu → hàm insert và truy vấn hoạt động đúng")
    logger.info("Nếu bảng rỗng → kiểm tra log lỗi ở trên (có thể do insert thất bại)")
    