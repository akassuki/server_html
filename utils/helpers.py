import json

def extract_numeric_values_and_keys(payload_str: str):
    try:
        payload = json.loads(payload_str)
        values = [v for v in payload.values() if isinstance(v, (int, float))]
        keys = [k for k, v in payload.items() if isinstance(v, (int, float))]
        return values, keys
    except Exception:
        return [], []
