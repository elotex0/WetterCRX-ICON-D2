import sys
import json
import cfgrib
import pandas as pd
import os
from zoneinfo import ZoneInfo
import numpy as np
import gc
import warnings
from encoding_config import NODATA_U8, LEVELS, ENCODING

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m', 'ww', 'tp', 'tp_acc', 'cape_ml', 'dbz_cmax', ...
os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]


def encode_uint8(data, enc):
    """float-Array (NaN erlaubt) -> uint8-Array (255 = NoData)."""
    nan_mask = ~np.isfinite(data)
    vmin, vmax, tr = enc["vmin"], enc["vmax"], enc["transform"]
    d = np.where(nan_mask, vmin, data).astype(np.float64)

    if tr == "raw":
        q = np.clip(np.rint(d), 0, LEVELS)
    else:
        d = np.clip(d, vmin, vmax)
        if tr == "linear":
            u = (d - vmin) / (vmax - vmin)
        elif tr == "sqrt":
            u = np.sqrt((d - vmin) / (vmax - vmin))
        elif tr == "ssqrt":
            u = (np.sign(d) * np.sqrt(np.abs(d) / vmax) + 1.0) / 2.0
        else:
            raise ValueError(tr)
        q = np.rint(u * LEVELS)

    q = q.astype(np.uint8)
    q[nan_mask] = NODATA_U8
    return q


# ------------------------------
# Bounding Box ICON-D2 (fix)
# ------------------------------
extent = [-3.94, 20.34, 43.18, 58.08]  # lon_min, lon_max, lat_min, lat_max


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

meta = {
    "var": var_type,
    "unit": enc["unit"],
    "dtype": "uint8",
    "nodata": NODATA_U8,
    "levels": LEVELS,
    "transform": enc["transform"],
    "vmin": enc["vmin"],
    "vmax": enc["vmax"],
    "decode": {
        "linear": "v = vmin + (q/254) * (vmax - vmin)",
        "sqrt": "v = vmin + (q/254)^2 * (vmax - vmin)",
        "ssqrt": "u = q/254*2 - 1; v = sign(u) * u^2 * vmax",
        "raw": "v = q",
    }[enc["transform"]],
    "orientation": "row0 = Norden, row-major, uint8",
    "projection": "EPSG:4326",
    "extent_4326": [extent[0], extent[2], extent[1], extent[3]],  # lon_min, lat_min, lon_max, lat_max
    "width": None,
    "height": None,
    "files": [],
}

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    valid_time_utc_override = None

    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn.lower() in ["ww", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
    elif var_type == "tp":
        tp_var = next((vn for vn in ["tp", "tot_prec"] if vn in ds), None)
        if tp_var is None:
            print(f"Keine Niederschlagsvariable in {filename}")
            ds.close()
            continue

        idx_now = all_files_global.index(filename)
        if idx_now + 1 >= len(all_files_global):
            print(f"{filename}: keine folgende Datei -> 1h-Niederschlag nicht berechenbar, überspringe")
            ds.close()
            continue

        next_path = os.path.join(data_dir, all_files_global[idx_now + 1])
        ds_next = cfgrib.open_dataset(next_path)

        if tp_var not in ds_next:
            print(f"Keine Niederschlagsvariable in {all_files_global[idx_now + 1]}")
            ds_next.close()
            ds.close()
            continue

        tp_now_vals = ds[tp_var].values
        tp_next_vals = ds_next[tp_var].values
        tp_now = tp_now_vals[0] if tp_now_vals.ndim == 3 else tp_now_vals
        tp_next = tp_next_vals[0] if tp_next_vals.ndim == 3 else tp_next_vals

        data = tp_next - tp_now

        vt_next_raw = ds_next["valid_time"].values
        valid_time_utc_override = pd.to_datetime(vt_next_raw[0]) if np.ndim(vt_next_raw) > 0 else pd.to_datetime(vt_next_raw)
        ds_next.close()
    elif var_type == "tp_acc":
        if "tp" not in ds:
            print(f"Keine tp-Variable in {filename}")
            ds.close()
            continue
        data = ds["tp"].isel(step=0).values
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds:
            print(f"Keine CAPE_ML-Variable in {filename}")
            ds.close()
            continue
        data = ds["CAPE_ML"].values[0, :, :]
        data[data < 0] = np.nan
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine passende Windvariable in {filename}")
            ds.close()
            continue
        data = ds["fg10"].values
        data[data < 0] = np.nan
        data = data * 3.6  # m/s -> km/h
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde-Variable in {filename}")
            ds.close()
            continue
        data = ds["sde"].values
        data[data < 0] = np.nan
        data = data * 100  # -> cm
    elif var_type == "change_snow":
        delta_hours = 6
        all_files = all_files_global
        filename_index = all_files.index(filename)

        snow_now = ds["sde"].values

        if filename_index == 0:
            data = np.full_like(snow_now, np.nan)
            print(f"{filename}: keine vorherige Datei -> Änderung = NaN")
        else:
            prev_index = max(0, filename_index - delta_hours)
            prev_file = os.path.join(data_dir, all_files[prev_index])
            ds_prev = cfgrib.open_dataset(prev_file)

            if "sde" not in ds_prev:
                print(f"Keine sde-Variable in {prev_file}")
                data = np.full_like(snow_now, np.nan)
            else:
                snow_prev = ds_prev["sde"].values
                snow_prev[snow_prev < 0] = np.nan
                data = (snow_now - snow_prev) * 100
                actual_hours = filename_index - prev_index
                print(f"{filename}: Schneehöhenänderung über {actual_hours}h berechnet")
            ds_prev.close()
    elif var_type == "cloud":
        if "CLCT" not in ds:
            print(f"Keine CLCT-Variable in {filename}")
            ds.close()
            continue
        data = ds["CLCT"].values
        data[data < 0] = np.nan
    elif var_type == "twater":
        if "TWATER" not in ds:
            print(f"Keine TWATER-Variable in {filename}")
            ds.close()
            continue
        data = ds["TWATER"].values
        data[data < 0] = np.nan
    elif var_type == "snowfall":
        if "SNOWLMT" not in ds:
            print(f"Keine SNOWLMT-Variable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["SNOWLMT"].values
        data[data < 0] = np.nan

    if data.ndim == 3:
        data = data[0]

    lon = ds["longitude"].values
    lat = ds["latitude"].values
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None

    if valid_time_utc_override is not None:
        valid_time_utc = valid_time_utc_override
    elif "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    # lat/lon aufsteigend sortieren (nötig für RegularGridInterpolator)
    if lon.ndim == 1 and lat.ndim == 1 and data.ndim == 2:
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[::-1, :]
        if lon[0] > lon[-1]:
            lon = lon[::-1]
            data = data[:, ::-1]

    # Gitter liegt nach der Sortierung oben schon Süden -> Norden vor.
    # row0 = Norden, dann nach uint8 quantisieren
    data_north_first = data[::-1]
    q = encode_uint8(data_north_first, enc)

    height, width = q.shape
    meta["width"], meta["height"] = int(width), int(height)

    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}"
    out_path = os.path.join(output_dir, outname)
    np.ascontiguousarray(q, dtype=np.uint8).tofile(out_path)

    meta["files"].append({
        "file": outname,
        "valid_time_utc": valid_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_local": valid_time_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    print(f"{filename} -> {outname} ({width}x{height}, {q.nbytes} Bytes)")

    ds.close()
    del data, data_north_first, q
    gc.collect()

print("META: " + json.dumps(meta, ensure_ascii=False))
