import requests
import random
import time
from datetime import datetime

SERVER_URL = "http://192.168.5.101:3000/api/data"

NODES = ["NODE_1","NODE_2","NODE_3"]

INTERVAL = 5 

def generate_sensor_data(node_id):
    return {
        "node":node_id,
        "temp": round(random.uniform(20.0,35.0),1),
        "humi": round(random.uniform(40.0,80.0),1),
        "no2" : round(random.uniform(10.0,40.0),1),
        "co2" : random.randint(400,1500),
        "co"  : round(random.uniform(0.0,5.0),1),
        "tvoc": random.randint(0,1000),
        "pm1_0": round(random.uniform(0.0,10.0),1),
        "pm2_5": round(random.uniform(0.0,20.0),1),
        "pm10": round(random.uniform(0.0,30.0),1),
    }

def send_node_data(node_id):
    payload = generate_sensor_data(node_id)
    try:
        response = requests.post(SERVER_URL,json=payload,timeout = 5)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {node_id} -> HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {node_id} -> {e}")

def main():
    print("=== PYTHON GATEWAY SIMULATOR (3 NODES) ===")
    print("Sending data to:",SERVER_URL)

    while True:
        for node in NODES:
            send_node_data(node)
            time.sleep(0.3)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()