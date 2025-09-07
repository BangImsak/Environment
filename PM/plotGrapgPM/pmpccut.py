import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import os
from datetime import datetime, timedelta

# =============== USER SETTINGS ===============
matplotlib.rcParams['font.family'] = 'Tahoma'

file_path     = r'/Users/lerritx/Downloads/PM_Comfort_2025-08-28.csv'        # ไฟล์ PC/PM
dht_file_path = r'/Users/lerritx/Downloads/DHT22_Comfort_2025-08-28.csv'     # ไฟล์ DHT22

png_output_dir = r'/Users/lerritx/ExportSignal/PM/Output/PNG'
csv_output_dir = r'/Users/lerritx/ExportSignal/PM/Output/CSV'

rolling_window_seconds = 5   # None = ปิด smoothing
# ============================================


# ---------- รับ section + เวลาเริ่ม ----------
user_str = input("กรอกค่า (เช่น  section1: Aug 28 2025 15:26:05.146 )\n> ").strip()
try:
    section_tag, dt_part = user_str.split(':', 1)
except ValueError:
    raise SystemExit("❌ รูปแบบไม่ถูกต้อง - ต้องมี ':' ขั้นระหว่าง section กับเวลา")

section_tag = section_tag.strip() or "sectionX"
start_dt    = pd.to_datetime(dt_part.strip(), dayfirst=False, utc=False,
                             errors='raise').to_pydatetime()
end_dt      = start_dt + timedelta(minutes=15)

# ใช้เฉพาะ 'เวลา' สำหรับกรองภายในวัน
start_time_str = start_dt.strftime('%H:%M:%S')
end_time_str   = end_dt.strftime('%H:%M:%S')
print(f"⏱  ช่วงเวลา : {start_time_str} – {end_time_str}  (เพิ่ม 15 นาทีอัตโนมัติ)")
print(f"🏷  section  : {section_tag}")

# ---------- helpers ----------
def parse_time_flex(s: str):
    s = s.replace('.', ':').replace('-', ':').replace('–', ':')
    return datetime.strptime(s, '%H:%M:%S').time()

def normalize_time_to_hms(t):
    if pd.isna(t): return t
    t = str(t).strip().replace('.', ':').replace('-', ':').replace('–', ':')
    if len(t.split(':')) == 2: t += ':00'
    return t

def make_timestamp(df):
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Time'] = df['Time'].astype(str).str.strip().apply(normalize_time_to_hms)
    ts = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                        errors='coerce', dayfirst=False)
    return (df.assign(timestamp=ts)
              .dropna(subset=['timestamp'])
              .sort_values('timestamp')
              .reset_index(drop=True))

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
# --------------------------------------------

# ---------- load files ----------
pm_df  = make_timestamp(pd.read_csv(file_path))
dht_df = make_timestamp(pd.read_csv(dht_file_path))

# detect DHT columns
dht_temp_cols = [c for c in dht_df.columns if c.lower().startswith('temp')]
dht_hum_cols  = [c for c in dht_df.columns if c.lower().startswith(('humid', 'rh'))]
dht_df  = ensure_numeric(dht_df, dht_temp_cols + dht_hum_cols)

# clip interval
pm_df['time_only'] = pm_df['timestamp'].dt.time
st, et = map(parse_time_flex, (start_time_str, end_time_str))
pm_clip = pm_df.loc[(pm_df['time_only'] >= st) & (pm_df['time_only'] <= et)] \
         if st <= et else pm_df.loc[(pm_df['time_only'] >= st) | (pm_df['time_only'] <= et)]

dht_clip = dht_df[(dht_df['timestamp'] >= pm_clip['timestamp'].min()) &
                  (dht_df['timestamp'] <= pm_clip['timestamp'].max())]

if pm_clip.empty or dht_clip.empty:
    raise SystemExit('⚠️ ไม่มีข้อมูลในช่วงเวลาที่ระบุ')

# merge
merged = pd.merge_asof(
    pm_clip.sort_values('timestamp'),
    dht_clip[['timestamp'] + dht_temp_cols + dht_hum_cols].sort_values('timestamp'),
    on='timestamp', direction='nearest', tolerance=pd.Timedelta('1s')
).set_index('timestamp')

# auto-groups
pc_cols   = [c for c in merged.columns if c.lower().startswith('pc')]
pm_cols   = [c for c in merged.columns if c.lower().startswith('pm')]
temp_cols = dht_temp_cols
hum_cols  = dht_hum_cols
num_cols  = pc_cols + pm_cols + temp_cols + hum_cols
plot_df   = merged[num_cols]

# smoothing
if rolling_window_seconds:
    plot_df = (plot_df
               .resample('1s').mean(numeric_only=True)
               .interpolate(limit=10)
               .rolling(rolling_window_seconds, min_periods=1).mean())

# ---------- output paths ----------
date_str = start_dt.strftime('%Y-%m-%d')

# แมป section → คำ
section_label_map = {
    'section1': 'Comfort',
    'section2': 'DisComfort',
    'section3': 'Recovery',
}
label = section_label_map.get(section_tag.lower(), 'Unknown')   # fallback ถ้าไม่ตรงแมป

base_name = f"PieraDHT22_{label}_{date_str}_{section_tag}"

os.makedirs(png_output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

png_path = os.path.join(png_output_dir, base_name + ".png")
csv_path = os.path.join(csv_output_dir, base_name + ".csv")

# ---------- plot ----------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

def pl(ax, cols, title, ylabel):
    if cols:
        for c in cols: ax.plot(plot_df.index, plot_df[c], label=c)
        ax.legend(fontsize='small', ncol=2)
    else:
        ax.text(.5, .5, f'ไม่พบ {title}', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True)

pl(axes[0], pc_cols, 'PC Counts', '#/cm³')
pl(axes[1], pm_cols, 'PM Concentrations', 'µg/m³')
pl(axes[2], temp_cols + hum_cols, 'Temperature / Humidity', '°C / %RH')

axes[-1].set_xlabel('เวลา (HH:MM:SS)')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------- save CSV ----------
plot_df = plot_df.round(3)

plot_df.reset_index().to_csv(
    csv_path,
    index=False,
    float_format='%.3f'           # บังคับให้เขียนออกเป็น 3 ตำแหน่ง (เช่น 1.230)
)
print(f"\n✅ PNG saved to: {png_path}")
print(f"✅ CSV saved to: {csv_path}")

# ---------- quick stats ----------
stats = plot_df.agg(['min','mean','max']).round(3).T
print("\nQuick stats (min / mean / max):")
print(stats)
