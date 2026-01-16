# test_spectral_fixed.py
import os
import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Config giả lập
CLUSTER_MIN_POINTS = 30
CLUSTER_MAX_CLUSTERS = 8
CLUSTER_USE_FULL_DAY = True
CLUSTER_MAX_POINTS = 1000
CLUSTER_DOWNSAMPLE_IF_EXCEED = 5

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def extract_numeric_values_and_keys(payload_str: str):
    try:
        payload = json.loads(payload_str)
        values = []
        keys = []
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                values.append(v)
                keys.append(k)
        return values, keys
    except:
        return [], []

# Hàm chính thức của bạn (copy nguyên)
def run_spectral_clustering_for_node(node: str):
    try:
        yesterday_start = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        yesterday_end = yesterday_start + timedelta(days=1) - timedelta(seconds=1)

        conn = sqlite3.connect("csdl_test_new.db")
        df = pd.read_sql_query(
            """
            SELECT payload, timestamp
            FROM measurements
            WHERE node = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
            """,
            conn,
            params=(node, yesterday_start, yesterday_end)
        )
        conn.close()

        if len(df) < CLUSTER_MIN_POINTS:
            logger.info(f"Không đủ dữ liệu clustering cho {node} (chỉ {len(df)}/{CLUSTER_MIN_POINTS})")
            return {"num_clusters": 0, "cluster_averages": []}

        if CLUSTER_USE_FULL_DAY:
            logger.info(f"Clustering cho {node}: dùng toàn bộ {len(df)} điểm ngày hôm trước")
            if len(df) > CLUSTER_MAX_POINTS:
                step = CLUSTER_DOWNSAMPLE_IF_EXCEED
                df = df.iloc[::step]
                logger.info(f"Downsample clustering: từ {len(df)*step} → {len(df)} điểm")
        else:
            df = df.tail(CLUSTER_MAX_POINTS)

        all_values = []
        timestamps = []
        feature_keys = None

        for _, row in df.iterrows():
            values, keys = extract_numeric_values_and_keys(row['payload'])
            if feature_keys is None:
                feature_keys = keys
            if len(values) != len(feature_keys):
                continue
            all_values.append(values)
            timestamps.append(row['timestamp'])

        if len(all_values) < CLUSTER_MIN_POINTS:
            return {"num_clusters": 0, "cluster_averages": []}

        data_array = np.array(all_values)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_array)

        best_n = 2
        best_score = -1.0
        max_possible = min(CLUSTER_MAX_CLUSTERS, len(scaled_data) // 5 + 1)
        for n in range(2, max_possible + 1):
            clustering_temp = SpectralClustering(n_clusters=n, affinity='rbf', random_state=42)
            labels_temp = clustering_temp.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels_temp)
            logger.debug(f"Thử n_clusters={n} cho {node}, Silhouette Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_n = n

        clustering = SpectralClustering(n_clusters=best_n, affinity='rbf', random_state=42)
        labels = clustering.fit_predict(scaled_data)
        logger.info(f"Spectral Clustering cho {node} - chọn {best_n} cụm (Silhouette Score tốt nhất: {best_score:.4f})")

        cluster_centroids = {}
        cluster_point_counts = {}
        for i, label in enumerate(labels):
            if label not in cluster_centroids:
                cluster_centroids[label] = np.zeros(len(feature_keys))
                cluster_point_counts[label] = 0
            cluster_centroids[label] += data_array[i]
            cluster_point_counts[label] += 1

        cluster_averages = []
        for label in cluster_centroids:
            centroid = cluster_centroids[label] / cluster_point_counts[label]
            avg_dict = {feature_keys[j]: round(float(centroid[j]), 2) for j in range(len(feature_keys))}
            cluster_averages.append({
                "cluster_id": int(label),
                "point_count": cluster_point_counts[label],
                "averages": avg_dict
            })

        cluster_summary = {}
        for i, label in enumerate(labels):
            ts_str = pd.to_datetime(timestamps[i]).strftime('%H:%M')
            cluster_summary.setdefault(label, []).append(ts_str)

        logger.info(f"Spectral Clustering cho {node} - {best_n} cụm:")
        for cid, times in sorted(cluster_summary.items()):
            times_str = ', '.join(sorted(times))[:120] + ('...' if len(times) > 20 else '')
            logger.info(f" Cụm {cid}: {len(times)} điểm → {times_str}")

        logger.info(f"Trung bình chỉ số từng cụm cho {node}:")
        for avg in cluster_averages:
            avg_str = ", ".join([f"{k}: {v}" for k, v in avg["averages"].items()])
            logger.info(f" Cụm {avg['cluster_id']}: {avg['point_count']} điểm → {avg_str}")

        cluster_date_str = yesterday_start.strftime('%Y-%m-%d')
        total_points = len(scaled_data)

        formatted_clusters = []
        for cl in cluster_averages:
            avgs = cl["averages"]
            formatted_clusters.append({
                "node": node,
                "cluster_date": cluster_date_str,
                "cluster_id": cl["cluster_id"],
                "point_count": cl["point_count"],
                "pm1_avg": avgs.get("pm1"),
                "pm25_avg": avgs.get("pm25"),
                "pm10_avg": avgs.get("pm10"),
                "co_avg": avgs.get("co"),
                "co2_avg": avgs.get("co2"),
                "temperature_avg": avgs.get("temperature"),
                "humidity_avg": avgs.get("humidity"),
                "tvoc_avg": avgs.get("tvoc"),
                "no2_avg": avgs.get("no2"),
            })

        return {
            "num_clusters": best_n,
            "silhouette_score": round(best_score, 4) if best_score > -1 else None,
            "total_points": total_points,
            "cluster_date": cluster_date_str,
            "cluster_averages": formatted_clusters
        }

    except Exception as e:
        logger.error(f"Lỗi spectral clustering cho {node}: {str(e)}")
        return {"num_clusters": 0, "cluster_averages": []}

# Tạo database test
def tao_csdl_test():
    db_name = "csdl_test_new.db"
    if os.path.exists(db_name):
        os.remove(db_name)

    conn = sqlite3.connect(db_name)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            payload TEXT NOT NULL
        )
    ''')

    ngay_hom_truoc = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    du_lieu_test = []

    # NODE_01: giá trị thấp
    np.random.seed(42)
    for i in range(100):
        payload = {
            "pm1": np.random.uniform(1.0, 5.0),
            "pm25": np.random.uniform(3.0, 12.0),
            "pm10": np.random.uniform(8.0, 25.0),
            "co": np.random.uniform(0.1, 0.4),
            "co2": np.random.uniform(300.0, 450.0),
            "temperature": np.random.uniform(20.0, 25.0),
            "humidity": np.random.uniform(50.0, 65.0),
            "tvoc": np.random.uniform(40.0, 100.0),
            "no2": np.random.uniform(4.0, 10.0)
        }
        thoi_gian = f"{ngay_hom_truoc} {(i//4):02d}:{(i%4)*15:02d}:00"
        du_lieu_test.append(("NODE_01", thoi_gian, json.dumps(payload)))

    # NODE_02: giá trị trung bình + nhiễu
    np.random.seed(100)
    for i in range(100):
        payload = {
            "pm1": np.random.normal(8.0, 3.0),
            "pm25": np.random.normal(18.0, 6.0),
            "pm10": np.random.normal(35.0, 12.0),
            "co": np.random.normal(0.5, 0.2),
            "co2": np.random.normal(550.0, 100.0),
            "temperature": np.random.normal(28.0, 4.0),
            "humidity": np.random.normal(70.0, 10.0),
            "tvoc": np.random.normal(150.0, 60.0),
            "no2": np.random.normal(15.0, 6.0)
        }
        thoi_gian = f"{ngay_hom_truoc} {(i//4):02d}:{(i%4)*15:02d}:00"
        du_lieu_test.append(("NODE_02", thoi_gian, json.dumps(payload)))

    # NODE_03: giá trị cao
    np.random.seed(200)
    for i in range(100):
        payload = {
            "pm1": np.random.uniform(10.0, 20.0),
            "pm25": np.random.uniform(20.0, 40.0),
            "pm10": np.random.uniform(40.0, 80.0),
            "co": np.random.uniform(0.7, 1.2),
            "co2": np.random.uniform(600.0, 900.0),
            "temperature": np.random.uniform(30.0, 40.0),
            "humidity": np.random.uniform(70.0, 90.0),
            "tvoc": np.random.uniform(200.0, 400.0),
            "no2": np.random.uniform(20.0, 40.0)
        }
        thoi_gian = f"{ngay_hom_truoc} {(i//4):02d}:{(i%4)*15:02d}:00"
        du_lieu_test.append(("NODE_03", thoi_gian, json.dumps(payload)))

    conn.executemany("INSERT INTO measurements (node, timestamp, payload) VALUES (?, ?, ?)", du_lieu_test)
    conn.commit()
    conn.close()
    logger.info("Đã tạo csdl test với 300 bản ghi (100 bản ghi mỗi node) với dữ liệu random độc lập")

# Chạy test
if __name__ == "__main__":
    tao_csdl_test()
    cac_node_test = ["NODE_01", "NODE_02", "NODE_03"]

    for node in cac_node_test:
        logger.info(f"\n=== BẮT ĐẦU TEST CHO NODE: {node} ===")
        ket_qua = run_spectral_clustering_for_node(node)

        logger.info(f"Kết quả cho {node}:")
        logger.info(f"- Số cụm: {ket_qua['num_clusters']}")
        logger.info(f"- Silhouette score: {ket_qua.get('silhouette_score')}")
        logger.info(f"- Total points: {ket_qua.get('total_points')}")
        logger.info(f"- Cluster date: {ket_qua.get('cluster_date')}")

        logger.info("- Chi tiết các cụm (định dạng cho DB):")
        for cum in ket_qua.get('cluster_averages', []):
            avg_str = ", ".join([f"{k}: {v}" for k, v in cum.items() if k not in ['node','cluster_date','cluster_id','point_count']])
            logger.info(f"  Cụm {cum['cluster_id']}: {cum['point_count']} điểm → {avg_str}")

        logger.info(f"=== KẾT THÚC TEST CHO NODE: {node} ===\n")

    print("\nTest hoàn tất. Kiểm tra log để xem chi tiết.")
    