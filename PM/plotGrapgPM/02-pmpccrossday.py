import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import os
import re
from datetime import datetime, date

# =================== ตั้งค่าผู้ใช้ ===================
matplotlib.rcParams['font.family'] = 'Tahoma'  # ฟอนต์ภาษาไทย (ถ้าไม่มีจะ fallback)

# --- ใส่ไฟล์ 1 หรือ 2 วันได้ ---
file_paths = [
    r'D:\The_Naeim\Data\PM\BackUpOtherUsing\PM_DisComfort_2025-09-06.csv',
    r'D:\The_Naeim\Data\PM\BackUpOtherUsing\PM_DisComfort_2025-09-07.csv',  # ถ้ามีใส่ได้เลย
]

# ถ้าใส่แค่ 1 ไฟล์ แล้วรูปแบบชื่อไฟล์มีวันที่ ระบบจะลองอนุมานไฟล์วันที่ปลายช่วงให้เอง
auto_infer_second_from_first = True

# ===== เลือกช่วง "วันที่ + เวลา" แบบต่อเนื่อง =====
# ตัวอย่าง: 27/08/2025 23:50:00  →  28/08/2025 00:20:00 (ข้ามเที่ยงคืน)
start_date_str = '06/09/2025'
start_time_str = '23:16:00'
end_date_str   = '07/09/2025'
end_time_str   = '09:35:00'  # ใช้ . หรือ - แทน : ก็ได้ เช่น 00-20-00

# โฟลเดอร์บันทึกรูป
output_base   = r'D:\The_Naeim\Data\PM\BackUpOtherUsing\outputpmpc'
custom_folder = os.path.join(output_base, 'custom_crossday')
os.makedirs(custom_folder, exist_ok=True)

# คอลัมน์ที่จะพล็อต (ถ้าไม่มีในไฟล์ โค้ดจะข้ามให้เอง)
pc_cols = ['PC0.1_calibrated']
pm_cols = ['PM2.5']

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
    # ลอง dayfirst ก่อน แล้วค่อย non-dayfirst
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
    ts = pd.to_datetime(dt_str, errors='coerce', dayfirst=True)
    miss = ts.isna()
    if miss.any():
        ts2 = pd.to_datetime(dt_str[miss], errors='coerce', dayfirst=False)
        ts.loc[miss] = ts2

    df['timestamp'] = ts
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

def replace_date_in_filename(path: str, target_date: date) -> str | None:
    """
    พยายามแทนที่วันที่ในชื่อไฟล์ด้วย target_date (รองรับรูปแบบ YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD)
    ถ้าหาไม่เจอรูปแบบวันที่ใน path จะคืน None
    """
    dirname, fname = os.path.split(path)
    ymd_dash = target_date.strftime('%Y-%m-%d')
    ymd_und  = target_date.strftime('%Y_%m_%d')
    ymd_raw  = target_date.strftime('%Y%m%d')

    patterns = [
        (re.compile(r'\d{4}-\d{2}-\d{2}'), ymd_dash),
        (re.compile(r'\d{4}_\d{2}_\d{2}'), ymd_und),
        (re.compile(r'\d{8}'), ymd_raw),
    ]
    new_fname = fname
    replaced = False
    for pat, repl in patterns:
        if pat.search(new_fname):
            new_fname = pat.sub(repl, new_fname, count=1)
            replaced = True
            break
    if not replaced:
        return None
    return os.path.join(dirname, new_fname)

# ---------- เตรียมช่วงเวลา ----------
start_dt = parse_datetime_pair(start_date_str, start_time_str)
end_dt   = parse_datetime_pair(end_date_str,   end_time_str)

# ถ้าเผลอใส่ปลายก่อนต้น สลับให้
if end_dt < start_dt:
    start_dt, end_dt = end_dt, start_dt

# ---------- โหลด CSV (1–2 ไฟล์) ----------
paths = list(file_paths)

# ถ้ามีแค่ 1 ไฟล์ และเปิด auto infer → สร้าง path ปลายช่วงจากวันที่ end_dt
if len(paths) == 1 and auto_infer_second_from_first:
    inferred = replace_date_in_filename(paths[0], end_dt.date())
    if inferred and inferred != paths[0] and os.path.exists(inferred):
        paths.append(inferred)

if len(paths) == 0:
    raise SystemExit("⛔ กรุณาระบุอย่างน้อย 1 ไฟล์ใน file_paths")

# อ่าน & รวม
dfs = []
for p in paths:
    if not os.path.exists(p):
        print(f"⚠️ ไม่พบไฟล์: {p} (ข้ามไฟล์นี้)")
        continue
    try:
        df_i = pd.read_csv(p)
        if not {'Date','Time'}.issubset(df_i.columns):
            print(f"⚠️ ไฟล์ {os.path.basename(p)} ไม่มีคอลัมน์ Date/Time (ข้าม)")
            continue
        df_i = build_timestamp(df_i)
        dfs.append(df_i)
    except Exception as e:
        print(f"⚠️ อ่านไฟล์ล้มเหลว {p}: {e}")

if not dfs:
    raise SystemExit("⛔ ไม่พบข้อมูลที่อ่านได้จากไฟล์ที่กำหนด")

df_all = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
if df_all.empty:
    raise SystemExit("⛔ ไม่มีข้อมูล timestamp ที่ใช้งานได้")

# ---------- ตัดช่วงเวลาแบบต่อเนื่อง (ข้ามวันได้) ----------
mask = (df_all['timestamp'] >= start_dt) & (df_all['timestamp'] <= end_dt)
period_df = df_all.loc[mask].copy()
if period_df.empty:
    raise SystemExit("⚠️ ไม่มีข้อมูลในช่วงที่กำหนด")

# ให้คอลัมน์ตัวเลขเป็น numeric
all_plot_cols = list(dict.fromkeys(pc_cols + pm_cols))
period_df = ensure_numeric(period_df, all_plot_cols)

# ตั้ง index เป็น timestamp เพื่อ resample/rolling ได้สะดวก
period_df = period_df.set_index('timestamp').sort_index()

# เลือกคอลัมน์ที่มีอยู่จริง
pc_cols_avail = [c for c in pc_cols if c in period_df.columns]
pm_cols_avail = [c for c in pm_cols if c in period_df.columns]

# ---------- (ออปชัน) ทำให้เรียบ: resample 1 วินาที + rolling ----------
plot_df = period_df.copy()
if rolling_window_seconds and (pc_cols_avail or pm_cols_avail):
    use_cols = pc_cols_avail + [c for c in pm_cols_avail if c not in pc_cols_avail]
    rs = plot_df[use_cols].resample('1S').mean().interpolate(limit=10)
    plot_df = rs.rolling(window=rolling_window_seconds, min_periods=1).mean()

# ---------- วาดกราฟ ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# PC (counts)
if pc_cols_avail:
    for col in pc_cols_avail:
        ax1.plot(plot_df.index, plot_df[col], label=col)
else:
    ax1.text(0.5, 0.5, "ไม่มีคอลัมน์ PC ที่ต้องการ", transform=ax1.transAxes,
             ha='center', va='center')
ax1.set_title(f'PC Counts ({start_dt.strftime("%d/%m %H:%M:%S")} – {end_dt.strftime("%d/%m %H:%M:%S")})')
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
ax2.set_title(f'PM Concentrations ({start_dt.strftime("%d/%m %H:%M:%S")} – {end_dt.strftime("%d/%m %H:%M:%S")})')
ax2.set_ylabel('ความเข้มข้น (µg/m³)')
ax2.legend(loc='upper right', ncol=2, fontsize='small')
ax2.grid(True)

# ฟอร์แมตแกนเวลา (ถ้าช่วงมีหลายวัน แสดงวันด้วย)
n_days = plot_df.index.normalize().nunique()
if n_days > 1:
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M:%S'))
else:
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

ax2.set_xlabel('เวลา')
plt.xticks(rotation=45)
plt.tight_layout()

# บันทึกรูป
def _stamp(dt: pd.Timestamp) -> str:
    return dt.strftime('%Y%m%d_%H%M%S')

filename = f"{_stamp(start_dt)}__{_stamp(end_dt)}_particle_data_crossday.png"
out_path = os.path.join(custom_folder, filename)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved cross-day plot: {out_path}")

# (ออปชัน) แสดงสถิติเร็ว ๆ
avail_cols = pc_cols_avail + [c for c in pm_cols_avail if c not in pc_cols_avail]
if avail_cols:
    stats = plot_df[avail_cols].agg(['min', 'mean', 'max']).round(3).T
    print("\nQuick stats (min / mean / max):")
    print(stats)
