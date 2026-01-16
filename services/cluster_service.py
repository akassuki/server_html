import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from config import CLUSTER_MIN_POINTS, CLUSTER_MAX_CLUSTERS, CLUSTER_USE_FULL_DAY, CLUSTER_MAX_POINTS, CLUSTER_DOWNSAMPLE_IF_EXCEED
from utils.helpers import extract_numeric_values_and_keys
from services.db_service import get_connection

logger = logging.getLogger("scheduler")


def run_spectral_clustering_for_node(node: str):
    """
    Phân cụm dữ liệu gần nhất của node bằng Spectral Clustering.
    Không lọc theo ngày, chỉ lấy bản ghi mới nhất.
    """
    try:
        conn = get_connection()
        # Lấy 300 bản ghi mới nhất (đủ để cluster, có thể tăng/giảm)
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

        if len(df) < CLUSTER_MIN_POINTS:
            logger.warning(f"Không đủ dữ liệu clustering cho {node} (chỉ {len(df)}/{CLUSTER_MIN_POINTS})")
            return {"num_clusters": 0, "cluster_averages": []}

        logger.info(f"Clustering cho {node}: dùng {len(df)} điểm gần nhất")

        # Giới hạn số điểm nếu vượt quá
        if len(df) > CLUSTER_MAX_POINTS:
            step = CLUSTER_DOWNSAMPLE_IF_EXCEED
            df = df.iloc[::step]
            logger.info(f"Downsample clustering: từ {len(df)*step} → {len(df)} điểm")
        # Không cần .tail() vì đã ORDER DESC rồi

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

        # Tìm số cụm tối ưu
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

        # Tính centroid
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

        # Log tóm tắt (giữ nguyên)
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

        # Trả về kết quả (không cần cluster_date, total_points nếu bạn không dùng)
        return {
            "num_clusters": best_n,
            "silhouette_score": round(best_score, 4) if best_score > -1 else None,
            "cluster_averages": cluster_averages  # giữ nguyên list dict với averages
        }

    except Exception as e:
        logger.error(f"Lỗi spectral clustering cho {node}: {str(e)}")
        return {"num_clusters": 0, "cluster_averages": []}
    