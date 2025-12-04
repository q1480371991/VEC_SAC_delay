from datetime import datetime
import time

if __name__ == "__main__":
    current_time = datetime.now()
    print(f"1. 完整时间: {current_time}")
    print(f"2. 格式化: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")