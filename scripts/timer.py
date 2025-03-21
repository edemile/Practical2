import time
import csv
import os
from functools import wraps

def timer(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            duration = end - start

            os.makedirs("timing_data", exist_ok=True)
            with open(f"timing_data/{name}_timing.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, duration])

            print(f"{name} took {duration:.4f}s")
            return result
        return wrapper
    return decorator
