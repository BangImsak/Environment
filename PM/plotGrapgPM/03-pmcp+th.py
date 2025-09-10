import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import os
import glob
import re
from datetime import datetime, date

# =================== ตั้งค่าผู้ใช้ ===================
matplotlib.rcParams['font.family'] = 'Tahoma'  # ฟอนต์ภาษาไทย (ถ้าไม่มีจะ fallback)

# 🔎 โฟลเดอร์แม่สำหรับค้นหาไฟล์ PM (ค้นหาแบบ recursive)
search_root_pm = r'D:\The_Naeim\Data\PM'    # เปลี่ยนเป็นโฟลเดอร์ของคุณ
search_recursive_pm = True                   # True = ค้นหาย่อยทั้งหมดด้วย

# 🌡️ โฟลเดอร์สำหรับค้นหาไฟล์ Temperature & Humidity
search_root_th = r'D:\The_Naeim\Data\DHT22'    # เปลี่ยนเป็นโฟลเดอร์ของคุณ
search_recursive_th = True                   # True = ค้นหาย่อยทั้งหมดด้วย

# ===== เลือกช่วง "วันที่ + เวลา" แบบต่อเนื่อง (ข้ามวันได้) =====
# ตัวอย่าง: 27/08/2025 23:50:00  →  28/08/2025 00:20:00
start_date_str = '10/09/2025'
start_time_str = '10:48:00'
end_date_str   = '10/09/2025'
end_time_str   = '13:10:00'  # ใช้ . หรือ - แทน : ได้ เช่น 00-20-00

# โฟลเดอร์บันทึกรูป
output_base   = r'D:\The_Naeim\Data\PM\BackUpOtherUsing\outputpmpc'
custom_folder = os.path.join(output_base, 'custom_crossday_auto')
os.makedirs(custom_folder, exist_ok=True)

# คอลัมน์ที่จะพล็อต (ถ้าไม่มีในไฟล์ โค้ดจะข้ามให้เอง)
pc_cols = ['PC0.1_calibrated']
pm_cols = ['PM2.5']
th_cols = ['Temperature', 'Humidity']  # คอลัมน์อุณหภูมิและความชื้น

# ทำเส้นให้เรียบขึ้น (วินาที) — ตั้งเป็น None เพื่อปิด
rolling_window_seconds = 5
# =================== จบการตั้งค่า ===================

def normalize_time_to_hms(s: str):
    """แปลงเวลาให้เป็น HH:MM:SS รองรับ HH:MM, HH.MM.SS, HH-MM-SS, HHMMSS, HHMM"""
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.replace('.', ':').replace('-', ':').replace('–', ':')
    if ':' not in s and s.isdigit():
        if len(s) == 6:   # HHMMSS
            s = f"{s[:2]}:{s[2:4]}:{s[4:]}"
        elif len(s) == 4: # HHMM
            s = f"{s[:2]}:{s[2:]}"
    if len(s.split(':')) == 2:  # HH:MM → HH:MM:00
        s = s + ':00'
    return s

def _to_datetime_flex(dt_str: str) -> pd.Timestamp:
    """พยายามแปลงสตริงวันที่-เวลาอย่างยืดหยุ่น (รองรับ / - . และ day-first)"""
    txt = dt_str.strip().replace('.', '/').replace('-', '/').replace('–', '/')
    try:
        return pd.to_datetime(txt, dayfirst=True, errors='raise')
    except Exception:
        return pd.to_datetime(txt, dayfirst=False, errors='raise')

def parse_datetime_pair(d_str: str, t_str: str) -> pd.Timestamp:
    return _to_datetime_flex(f"{d_str.strip()} {normalize_time_to_hms(t_str)}")

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างคอลัมน์ timestamp ที่ robust จาก Date + Time (รองรับหลายรูปแบบวัน/เวลา)"""
    missing_tokens = {'', 'nan', 'NaN', 'NaT', 'None', 'NULL', 'null'}
    # ทำความสะอาด
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Time'] = df['Time'].astype(str).str.strip()
    df['Date'] = df['Date'].mask(df['Date'].str.lower().isin(missing_tokens)).ffill()
    df['Time'] = df['Time'].mask(df['Time'].str.lower().isin(missing_tokens))
    df['Time'] = df['Time'].apply(normalize_time_to_hms)

    dt_str = (df['Date'] + ' ' + df['Time']).str.strip()
    # แก้ไข: ใช้ dayfirst=False เพื่อ parsing YYYY/MM/DD จาก CSV ได้ถูกต้อง
    ts = pd.to_datetime(dt_str, errors='coerce', dayfirst=False)
    df['timestamp'] = ts
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

def find_csv_for_date(target_date: date, root: str, recursive: bool = True) -> str | None:
    """
    ค้นหาไฟล์ตามแพทเทิร์น *_<YYYY-MM-DD>.csv ภายใต้โฟลเดอร์ root
    - ถ้าพบหลายไฟล์: เลือก 'ไฟล์ที่แก้ไขล่าสุด' (mtime มากสุด)
    - ไม่พบ: คืน None
    """
    ymd = target_date.strftime('%Y-%m-%d')
    pattern = f"*_{ymd}.csv"
    root = os.path.expanduser(root)

    if recursive:
        glob_pattern = os.path.join(root, "**", pattern)
        matches = glob.glob(glob_pattern, recursive=True)
    else:
        glob_pattern = os.path.join(root, pattern)
        matches = glob.glob(glob_pattern)

    if not matches:
        print(f"⚠️ ไม่พบไฟล์สำหรับวันที่ {ymd} ภายใต้ {root} (pattern: {pattern})")
        return None

    # เลือกไฟล์ที่แก้ไขล่าสุด
    best = max(matches, key=lambda p: os.path.getmtime(p))
    print(f"✅ เลือกไฟล์ {os.path.basename(best)} สำหรับวันที่ {ymd}")
    return best

def load_data_for_daterange(root_folder: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, 
                           recursive: bool = True, data_type: str = "data") -> pd.DataFrame:
    """โหลดข้อมูลจากไฟล์ CSV ในช่วงวันที่ที่กำหนด"""
    all_days = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq='D')
    paths = []
    
    for d in all_days.date:
        p = find_csv_for_date(d, root_folder, recursive=recursive)
        if p and p not in paths:
            paths.append(p)
    
    if not paths:
        print(f"⚠️ ไม่พบไฟล์ {data_type} ตามแพทเทิร์นสำหรับทุกวันในช่วงที่กำหนด")
        return pd.DataFrame()
    
    dfs = []
    for p in paths:
        try:
            df_i = pd.read_csv(p)
            if not {'Date','Time'}.issubset(df_i.columns):
                print(f"⚠️ ไฟล์ {os.path.basename(p)} ไม่มีคอลัมน์ Date/Time (ข้าม)")
                continue
            df_i = build_timestamp(df_i)
            dfs.append(df_i)
        except Exception as e:
            print(f"⚠️ อ่านไฟล์ {data_type} ล้มเหลว {p}: {e}")
    
    if not dfs:
        print(f"⚠️ ไม่พบข้อมูล {data_type} ที่อ่านได้จากไฟล์ที่ค้นพบ")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp')

# ---------- เตรียมช่วงเวลา ----------
start_dt = parse_datetime_pair(start_date_str, start_time_str)
end_dt   = parse_datetime_pair(end_date_str,   end_time_str)

# ถ้าเผลอใส่ปลายก่อนต้น สลับให้
if end_dt < start_dt:
    start_dt, end_dt = end_dt, start_dt

print(f"📅 ช่วงเวลา: {start_dt.strftime('%d/%m/%Y %H:%M:%S')} → {end_dt.strftime('%d/%m/%Y %H:%M:%S')}")

# ---------- โหลดข้อมูล PM ----------
print("\n🔍 กำลังโหลดข้อมูล PM...")
df_pm = load_data_for_daterange(search_root_pm, start_dt, end_dt, search_recursive_pm, "PM")

# ---------- โหลดข้อมูล Temperature & Humidity ----------
print("\n🌡️ กำลังโหลดข้อมูล Temperature & Humidity...")
df_th = load_data_for_daterange(search_root_th, start_dt, end_dt, search_recursive_th, "T&H")

# ตรวจสอบว่ามีข้อมูลอย่างน้อย 1 ชุด
if df_pm.empty and df_th.empty:
    raise SystemExit("⛔ ไม่พบข้อมูลที่อ่านได้จากทุกไฟล์")

# ---------- ตัดช่วงเวลาและรวมข้อมูล ----------
combined_data = []

if not df_pm.empty:
    mask_pm = (df_pm['timestamp'] >= start_dt) & (df_pm['timestamp'] <= end_dt)
    period_pm = df_pm.loc[mask_pm].copy()
    if not period_pm.empty:
        period_pm = ensure_numeric(period_pm, pc_cols + pm_cols)
        combined_data.append(period_pm)

if not df_th.empty:
    mask_th = (df_th['timestamp'] >= start_dt) & (df_th['timestamp'] <= end_dt)
    period_th = df_th.loc[mask_th].copy()
    if not period_th.empty:
        period_th = ensure_numeric(period_th, th_cols)
        combined_data.append(period_th)

if not combined_data:
    raise SystemExit("⚠️ ไม่มีข้อมูลในช่วงที่กำหนด")

# รวมข้อมูลโดยใช้ outer join บน timestamp
if len(combined_data) == 1:
    period_df = combined_data[0]
else:
    # รวมข้อมูล PM และ T&H
    df1 = combined_data[0].set_index('timestamp')
    df2 = combined_data[1].set_index('timestamp')
    period_df = df1.join(df2, how='outer', rsuffix='_th').sort_index()
    period_df = period_df.reset_index()

# ตั้ง index เป็น timestamp เพื่อ resample/rolling ได้สะดวก
period_df = period_df.set_index('timestamp').sort_index()

# เลือกคอลัมน์ที่มีอยู่จริง
pc_cols_avail = [c for c in pc_cols if c in period_df.columns]
pm_cols_avail = [c for c in pm_cols if c in period_df.columns]
th_cols_avail = [c for c in th_cols if c in period_df.columns]

print(f"\n📊 คอลัมน์ที่พบ:")
print(f"   PC: {pc_cols_avail}")
print(f"   PM: {pm_cols_avail}")
print(f"   T&H: {th_cols_avail}")

# ---------- (ออปชัน) ทำให้เรียบ: resample 1 วินาที + rolling ----------
plot_df = period_df.copy()
if rolling_window_seconds:
    all_cols = pc_cols_avail + pm_cols_avail + th_cols_avail
    if all_cols:
        rs = plot_df[all_cols].resample('1S').mean().interpolate(limit=10)
        plot_df = rs.rolling(window=rolling_window_seconds, min_periods=1).mean()

# ---------- วาดกราห 4 แผง ----------
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

# แผง 1: PC Counts + Temperature
if pc_cols_avail:
    color1 = 'tab:blue'
    ax1.set_ylabel('จำนวนอนุภาค (#/cm³)', color=color1)
    for col in pc_cols_avail:
        ax1.plot(plot_df.index, plot_df[col], color=color1, label=col)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper left', fontsize='small')
else:
    ax1.text(0.5, 0.5, "ไม่มีคอลัมน์ PC", transform=ax1.transAxes, ha='center', va='center')

if 'Temperature' in th_cols_avail:
    ax1_temp = ax1.twinx()
    color_temp = 'tab:red'
    ax1_temp.set_ylabel('อุณหภูมิ (°C)', color=color_temp)
    ax1_temp.plot(plot_df.index, plot_df['Temperature'], color=color_temp, label='Temperature', alpha=0.7)
    ax1_temp.tick_params(axis='y', labelcolor=color_temp)
    ax1_temp.legend(loc='upper right', fontsize='small')

ax1.set_title(f'PC Counts & Temperature ({start_dt.strftime("%d/%m %H:%M:%S")} – {end_dt.strftime("%d/%m %H:%M:%S")})')
ax1.grid(True, alpha=0.3)

# แผง 2: PC Counts + Humidity
if pc_cols_avail:
    color1 = 'tab:blue'
    ax2.set_ylabel('จำนวนอนุภาค (#/cm³)', color=color1)
    for col in pc_cols_avail:
        ax2.plot(plot_df.index, plot_df[col], color=color1, label=col)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.legend(loc='upper left', fontsize='small')
else:
    ax2.text(0.5, 0.5, "ไม่มีคอลัมน์ PC", transform=ax2.transAxes, ha='center', va='center')

if 'Humidity' in th_cols_avail:
    ax2_hum = ax2.twinx()
    color_hum = 'tab:green'
    ax2_hum.set_ylabel('ความชื้น (%RH)', color=color_hum)
    ax2_hum.plot(plot_df.index, plot_df['Humidity'], color=color_hum, label='Humidity', alpha=0.7)
    ax2_hum.tick_params(axis='y', labelcolor=color_hum)
    ax2_hum.legend(loc='upper right', fontsize='small')

ax2.set_title(f'PC Counts & Humidity ({start_dt.strftime("%d/%m %H:%M:%S")} – {end_dt.strftime("%d/%m %H:%M:%S")})')
ax2.grid(True, alpha=0.3)

# แผง 3: PM + Temperature
if pm_cols_avail:
    color2 = 'tab:orange'
    ax3.set_ylabel('ความเข้มข้น (µg/m³)', color=color2)
    for col in pm_cols_avail:
        ax3.plot(plot_df.index, plot_df[col], color=color2, label=col)
    ax3.tick_params(axis='y', labelcolor=color2)
    ax3.legend(loc='upper left', fontsize='small')
else:
    ax3.text(0.5, 0.5, "ไม่มีคอลัมน์ PM", transform=ax3.transAxes, ha='center', va='center')

if 'Temperature' in th_cols_avail:
    ax3_temp = ax3.twinx()
    color_temp = 'tab:red'
    ax3_temp.set_ylabel('อุณหภูมิ (°C)', color=color_temp)
    ax3_temp.plot(plot_df.index, plot_df['Temperature'], color=color_temp, label='Temperature', alpha=0.7)
    ax3_temp.tick_params(axis='y', labelcolor=color_temp)
    ax3_temp.legend(loc='upper right', fontsize='small')

ax3.set_title(f'PM Concentrations & Temperature ({start_dt.strftime("%d/%m %H:%M:%S")} – {end_dt.strftime("%d/%m %H:%M:%S")})')
ax3.grid(True, alpha=0.3)

# แผง 4: PM + Humidity
if pm_cols_avail:
    color2 = 'tab:orange'
    ax4.set_ylabel('ความเข้มข้น (µg/m³)', color=color2)
    for col in pm_cols_avail:
        ax4.plot(plot_df.index, plot_df[col], color=color2, label=col)
    ax4.tick_params(axis='y', labelcolor=color2)
    ax4.legend(loc='upper left', fontsize='small')
else:
    ax4.text(0.5, 0.5, "ไม่มีคอลัมน์ PM", transform=ax4.transAxes, ha='center', va='center')

if 'Humidity' in th_cols_avail:
    ax4_hum = ax4.twinx()
    color_hum = 'tab:green'
    ax4_hum.set_ylabel('ความชื้น (%RH)', color=color_hum)
    ax4_hum.plot(plot_df.index, plot_df['Humidity'], color=color_hum, label='Humidity', alpha=0.7)
    ax4_hum.tick_params(axis='y', labelcolor=color_hum)
    ax4_hum.legend(loc='upper right', fontsize='small')

ax4.set_title(f'PM Concentrations & Humidity ({start_dt.strftime("%d/%m %H:%M:%S")} – {end_dt.strftime("%d/%m %H:%M:%S")})')
ax4.grid(True, alpha=0.3)

# ฟอร์แมตแกนเวลา (ถ้าช่วงมีหลายวัน แสดงวันด้วย)
n_days = plot_df.index.normalize().nunique()
time_format = '%d/%m %H:%M:%S' if n_days > 1 else '%H:%M:%S'

for ax in [ax2, ax4]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    ax.set_xlabel('เวลา')

plt.xticks(rotation=45)
plt.tight_layout()

# บันทึกรูป
def _stamp(dt: pd.Timestamp) -> str:
    return dt.strftime('%Y%m%d_%H%M%S')

filename = f"{_stamp(start_dt)}__{_stamp(end_dt)}_particle_data_with_th_auto.png"
out_path = os.path.join(custom_folder, filename)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved enhanced plot with T&H: {out_path}")

# (ออปชัน) แสดงสถิติเร็ว ๆ
avail_cols = pc_cols_avail + pm_cols_avail + th_cols_avail
if avail_cols:
    stats = plot_df[avail_cols].agg(['min', 'mean', 'max']).round(3).T
    print("\nQuick stats (min / mean / max):")
    print(stats)