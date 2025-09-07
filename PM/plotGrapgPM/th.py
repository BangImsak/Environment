import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import os
from datetime import datetime

# =================== ตั้งค่าผู้ใช้ ===================
matplotlib.rcParams['font.family'] = 'Tahoma'  # ฟอนต์ภาษาไทย (ถ้าไม่มีจะ fallback)

# ชี้ไปที่ไฟล์ DHT22 ของคุณ (หัวคอลัมน์: Date, Time, Temperature, Humidity)
file_path = r'/Users/lerritx/Downloads/DHT22_2025-08-13.csv'

# กำหนดช่วงเวลา (รองรับ HH:MM หรือ HH:MM:SS รวมถึง 15.12.47, 15-12-47, 151247)
start_time_str = '00.00.00'
end_time_str   = '23.59.59'

# โฟลเดอร์บันทึกรูป
output_base   = r'/Users/lerritx/ExportSignal/TH'
custom_folder = os.path.join(output_base, 'custom')
os.makedirs(custom_folder, exist_ok=True)

# ทำเส้นให้เรียบขึ้น (วินาที) — ตั้งเป็น None เพื่อปิด
rolling_window_seconds = 5
# =================== จบการตั้งค่า ===================

def load_temp_humid_df_th(path, rolling_window_seconds=5):
    """
    อ่านไฟล์ DHT22 ตามสเปค th.py (Date, Time, Temperature, Humidity)
    คืนค่า DataFrame คอลัมน์: Timestamp, Temperature, Humidity
    resample 1s + optional rolling smoothing
    """
    df = pd.read_csv(path)

    required_cols = {'Date', 'Time', 'Temperature', 'Humidity'}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ไฟล์ DHT22 ขาดคอลัมน์: {missing}")

    # ทำความสะอาดเวลาตาม th.py
    missing_tokens = {'', 'nan', 'NaN', 'NaT', 'None', 'NULL', 'null'}
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Time'] = df['Time'].astype(str).str.strip()
    df['Date'] = df['Date'].mask(df['Date'].str.lower().isin(missing_tokens)).ffill()
    df['Time'] = df['Time'].mask(df['Time'].str.lower().isin(missing_tokens))
    df['Time'] = df['Time'].apply(normalize_time_to_hms)

    dt_str = (df['Date'] + ' ' + df['Time']).str.strip()
    ts = pd.to_datetime(dt_str, errors='coerce', dayfirst=True)
    missing = ts.isna()
    if missing.any():
        ts2 = pd.to_datetime(dt_str[missing], errors='coerce', dayfirst=False)
        ts.loc[missing] = ts2

    df['Timestamp'] = ts
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

    # numeric + resample 1s + rolling (optional)
    for c in ['Temperature', 'Humidity']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = (
        df[['Timestamp', 'Temperature', 'Humidity']]
        .set_index('Timestamp')
        .resample('1S').mean()
        .interpolate(limit=10)
    )
    if rolling_window_seconds:
        out = out.rolling(window=rolling_window_seconds, min_periods=1).mean()

    return out.reset_index()

def parse_time_flex(s: str):
    """รองรับ HH:MM[:SS], HH.MM[.SS], HH-MM[-SS], HHMMSS, HHMM"""
    s = str(s).strip()
    s_norm = s.replace('.', ':').replace('-', ':').replace('–', ':')
    if s_norm.isdigit():
        if len(s_norm) == 6:   # HHMMSS
            s_norm = f"{s_norm[0:2]}:{s_norm[2:4]}:{s_norm[4:6]}"
        elif len(s_norm) == 4: # HHMM
            s_norm = f"{s_norm[0:2]}:{s_norm[2:4]}"
    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            return datetime.strptime(s_norm, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"ไม่สามารถแปลงเวลาได้: {s}")

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def normalize_time_to_hms(s: str):
    """แปลงเวลาให้เป็น HH:MM:SS หากเป็น HH:MM หรือรูปแบบจุด/ขีด/เลขล้วน"""
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.replace('.', ':').replace('-', ':').replace('–', ':')
    if ':' not in s and s.isdigit():
        if len(s) == 6:
            s = f"{s[:2]}:{s[2:4]}:{s[4:]}"
        elif len(s) == 4:
            s = f"{s[:2]}:{s[2:]}"
    if len(s.split(':')) == 2:
        s = s + ':00'
    return s

# ---------- โหลดและเตรียมข้อมูล ----------
df = pd.read_csv(file_path)

required_cols = {'Date', 'Time', 'Temperature', 'Humidity'}
missing_req = [c for c in required_cols if c not in df.columns]
if missing_req:
    raise SystemExit(f"⛔ ไฟล์ไม่มีคอลัมน์ที่ต้องใช้: {missing_req}")

missing_tokens = {'', 'nan', 'NaN', 'NaT', 'None', 'NULL', 'null'}
df['Date'] = df['Date'].astype(str).str.strip()
df['Time'] = df['Time'].astype(str).str.strip()
df['Date'] = df['Date'].mask(df['Date'].str.lower().isin(missing_tokens)).ffill()
df['Time'] = df['Time'].mask(df['Time'].str.lower().isin(missing_tokens))
df['Time'] = df['Time'].apply(normalize_time_to_hms)

dt_str = (df['Date'] + ' ' + df['Time']).str.strip()
ts = pd.to_datetime(dt_str, errors='coerce', dayfirst=True)
missing = ts.isna()
if missing.any():
    ts2 = pd.to_datetime(dt_str[missing], errors='coerce', dayfirst=False)
    ts.loc[missing] = ts2

df['timestamp'] = ts
df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
if df.empty:
    raise SystemExit("⛔ ไม่พบ timestamp ที่พอใช้ได้ในไฟล์")

df['time_only'] = df['timestamp'].dt.time

start_t = parse_time_flex(start_time_str)
end_t   = parse_time_flex(end_time_str)

if start_t <= end_t:
    mask = (df['time_only'] >= start_t) & (df['time_only'] <= end_t)
else:
    mask = (df['time_only'] >= start_t) | (df['time_only'] <= end_t)

period_df = df.loc[mask].copy()
if period_df.empty:
    raise SystemExit("⚠️ ไม่มีข้อมูลในช่วงเวลาที่กำหนด")

use_cols = ['Temperature', 'Humidity']
period_df = ensure_numeric(period_df, use_cols)
period_df = period_df.set_index('timestamp')

plot_df = period_df.copy()
if rolling_window_seconds and use_cols:
    rs = plot_df[use_cols].resample('1S').mean().interpolate(limit=10)
    plot_df = rs.rolling(window=rolling_window_seconds, min_periods=1).mean()

# ---------- วาดกราฟ (รวมแกนซ้าย–ขวา) ----------
fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))

# Temperature (แกนซ้าย)
line_temp, = ax1.plot(plot_df.index, plot_df['Temperature'], label='Temperature', linestyle='-')
ax1.set_title(f'Temperature & Humidity ({start_time_str}–{end_time_str} น.)')
ax1.set_ylabel('อุณหภูมิ (°C)')
ax1.grid(True)

# Humidity (แกนขวา) — เส้นตรงสีแดง
ax2 = ax1.twinx()
line_hum, = ax2.plot(
    plot_df.index,
    plot_df['Humidity'],
    label='Humidity',
    linestyle='-',      # ← เปลี่ยนเป็นเส้นตรง (solid)
    color='red'
)
ax2.set_ylabel('ความชื้นสัมพัทธ์ (%)')

# รวม legend
lines = [line_temp, line_hum]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize='small')

# แกนเวลา
ax1.set_xlabel('เวลา (HH:MM:SS)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)
plt.tight_layout()

# บันทึกรูป
def _sanitize_time_for_filename(s: str):
    return (s.replace(':','').replace('.','').replace('-','').replace('–',''))

filename = f"{_sanitize_time_for_filename(start_time_str)}_{_sanitize_time_for_filename(end_time_str)}_temp_hum_dualaxis_redhum_solid.png"
out_path = os.path.join(custom_folder, filename)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved combined Temp/Humidity plot: {out_path}")

# (ออปชัน) สถิติเร็ว ๆ
stats = plot_df[use_cols].agg(['min', 'mean', 'max']).round(3).T
print("\nQuick stats (min / mean / max):")
print(stats)
