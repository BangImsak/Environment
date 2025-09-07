import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import os
from datetime import datetime

# =================== ตั้งค่าผู้ใช้ ===================
matplotlib.rcParams['font.family'] = 'Tahoma'  # ฟอนต์ภาษาไทย (ถ้าไม่มีจะ fallback)

# ชี้ไปที่ไฟล์ของคุณ (ใช้ไฟล์ที่แก้ Date เป็น DD/MM/YYYY แล้วจะดีที่สุด)
file_path = r'D:\The_Naeim\Data\PM\BackUpOtherUsing\PM_DisComfort_2025-09-06.csv'   # หรือ ..._FIXED.csv

# กำหนดช่วงเวลา (รองรับ HH:MM หรือ HH:MM:SS รวมถึง 15.12.47, 15-12-47, 151247)
start_time_str = '18.44.00'
end_time_str   = '23.16.00'

#15.12.47 - 15.23.47
#15.42.40 - 15-53.40
#16.27.11 - 16.38.11

# โฟลเดอร์บันทึกรูป
output_base   = r'D:\The_Naeim\Data\PM\BackUpOtherUsing\PunSukToSafeCount1'
custom_folder = os.path.join(output_base, 'custom')
os.makedirs(custom_folder, exist_ok=True)

# คอลัมน์ที่จะพล็อต (ถ้าไม่มีบางคอลัมน์ในไฟล์ โค้ดจะข้ามให้เอง)
pc_cols = ['PC0.1_calibrated']
pm_cols = ['PM2.5']

# ทำเส้นให้เรียบขึ้น (วินาที) — ตั้งเป็น None เพื่อปิด
rolling_window_seconds = 5
# =================== จบการตั้งค่า ===================

def parse_time_flex(s: str):
    """รองรับ HH:MM[:SS], HH.MM[.SS], HH-MM[-SS], HHMMSS, HHMM"""
    s = str(s).strip()
    s_norm = s.replace('.', ':').replace('-', ':').replace('–', ':')
    if s_norm.isdigit():  # เช่น 151247 หรือ 1512
        if len(s_norm) == 6:   # HHMMSS
            s_norm = f"{s_norm[0:2]}:{s_norm[2:4]}:{s_norm[4:6]}"
        elif len(s_norm) == 4: # HHMM
            s_norm = f"{s_norm[0:2]}:{s_norm[2:4]}"
    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            return datetime.strptime(s_norm, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"ไม่สามารถแปลงเวลาได้: {s} (รองรับ HH:MM[:SS], HH.MM[.SS], HH-MM[-SS], HHMMSS)")

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
        if len(s) == 6:   # HHMMSS
            s = f"{s[:2]}:{s[2:4]}:{s[4:]}"
        elif len(s) == 4: # HHMM
            s = f"{s[:2]}:{s[2:]}"
    if len(s.split(':')) == 2:
        s = s + ':00'
    return s

# ---------- โหลดและเตรียมข้อมูล ----------
df = pd.read_csv(file_path)

# ทำความสะอาดคอลัมน์ Date/Time
missing_tokens = {'', 'nan', 'NaN', 'NaT', 'None', 'NULL', 'null'}
df['Date'] = df['Date'].astype(str).str.strip()
df['Time'] = df['Time'].astype(str).str.strip()
df['Date'] = df['Date'].mask(df['Date'].str.lower().isin(missing_tokens)).ffill()
df['Time'] = df['Time'].mask(df['Time'].str.lower().isin(missing_tokens))
df['Time'] = df['Time'].apply(normalize_time_to_hms)

# สร้าง timestamp แบบ robust: ลอง dayfirst=True ก่อน แล้วค่อย False
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

# คอลัมน์เวลาล้วน
df['time_only'] = df['timestamp'].dt.time

# แปลงช่วงเวลา (รองรับวินาที)
start_t = parse_time_flex(start_time_str)
end_t   = parse_time_flex(end_time_str)

# กรองช่วงเวลา — รองรับการข้ามเที่ยงคืน
if start_t <= end_t:
    mask = (df['time_only'] >= start_t) & (df['time_only'] <= end_t)
else:
    mask = (df['time_only'] >= start_t) | (df['time_only'] <= end_t)

period_df = df.loc[mask].copy()
if period_df.empty:
    raise SystemExit("⚠️ ไม่มีข้อมูลในช่วงเวลาที่กำหนด")

# ให้คอลัมน์ตัวเลขเป็น numeric
all_plot_cols = list(dict.fromkeys(pc_cols + pm_cols))
period_df = ensure_numeric(period_df, all_plot_cols)

# ตั้ง index เป็น timestamp เพื่อ resample/rolling ได้สะดวก
period_df = period_df.set_index('timestamp')

# เลือกคอลัมน์ที่มีอยู่จริง
pc_cols_avail = [c for c in pc_cols if c in period_df.columns]
pm_cols_avail = [c for c in pm_cols if c in period_df.columns]

# (ออปชัน) ทำให้เรียบ: resample 1 วินาที + rolling
plot_df = period_df.copy()
if rolling_window_seconds and (pc_cols_avail or pm_cols_avail):
    use_cols = pc_cols_avail + [c for c in pm_cols_avail if c not in pc_cols_avail]
    rs = plot_df[use_cols].resample('1S').mean().interpolate(limit=10)
    plot_df = rs.rolling(window=rolling_window_seconds, min_periods=1).mean()

# ---------- วาดกราฟ ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# PC (counts)
if pc_cols_avail:
    for col in pc_cols_avail:
        ax1.plot(plot_df.index, plot_df[col], label=col)
else:
    ax1.text(0.5, 0.5, "ไม่มีคอลัมน์ PC ที่ต้องการ", transform=ax1.transAxes,
             ha='center', va='center')
ax1.set_title(f'PC Counts ({start_time_str}–{end_time_str} น.)')
ax1.set_ylabel('จำนวนอนุภาค (#/cm³)')
ax1.legend(loc='upper right', ncol=2, fontsize='small')
ax1.grid(True)

# PM (µg/m³)
if pm_cols_avail:
    for col in pm_cols_avail:
        ax2.plot(plot_df.index, plot_df[col], label=col)
else:
    ax2.text(0.5, 0.5, "ไม่มีคอลัมน์ PM ที่ต้องการ", transform=ax2.transAxes,
             ha='center', va='center')
ax2.set_title(f'PM Concentrations ({start_time_str}–{end_time_str} น.)')
ax2.set_ylabel('ความเข้มข้น (µg/m³)')
ax2.legend(loc='upper right', ncol=2, fontsize='small')
ax2.grid(True)

# แกนเวลาแสดง "วินาที"
ax2.set_xlabel('เวลา (HH:MM:SS)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)
plt.tight_layout()

# บันทึกรูป
def _sanitize_time_for_filename(s: str):
    return (s.replace(':','').replace('.','').replace('-','').replace('–',''))

filename = f"{_sanitize_time_for_filename(start_time_str)}_{_sanitize_time_for_filename(end_time_str)}_particle_data_timeonly.png"
out_path = os.path.join(custom_folder, filename)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved time-only plot: {out_path}")

# (ออปชัน) แสดงสถิติเร็ว ๆ
avail_cols = pc_cols_avail + [c for c in pm_cols_avail if c not in pc_cols_avail]
if avail_cols:
    stats = plot_df[avail_cols].agg(['min', 'mean', 'max']).round(3).T
    print("\nQuick stats (min / mean / max):")
    print(stats)
