#!/usr/bin/env bash
cd /d/The_Naeim/Data   # D:\The_Naeim\Data

git add .
git commit -m "auto push $(date '+%Y-%m-%d %H:%M:%S')" --allow-empty
git push origin main   # ถ้า branch เป็น main
