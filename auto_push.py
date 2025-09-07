import os
import subprocess
import schedule
import time
from datetime import datetime

def git_push():
    # วันที่ในรูปแบบ วัน-เดือน-ปี (เช่น 07-09-2025)
    commit_message = datetime.now().strftime("%d-%m-%Y")

    try:
        # git add .
        subprocess.run(["git", "add", "."], check=True)

        # git commit -m "วันที่"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # git push -u origin main
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

        print(f"[{datetime.now()}] Push success with commit message: {commit_message}")

    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Error: {e}")

# ตั้งเวลาให้รันทุกวันตอน 00:00
schedule.every().day.at("00:00").do(git_push)

print("Git auto-push service started. Waiting for midnight...")

# loop รอให้ถึงเวลา
while True:
    schedule.run_pending()
    time.sleep(30)  # เช็คทุก 30 วินาที
