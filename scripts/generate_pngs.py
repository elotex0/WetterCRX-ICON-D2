import sys
import cfgrib
import pandas as pd
import os
from zoneinfo import ZoneInfo
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import gc
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from PIL import Image
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m', 'ww', 'tp', 'tp_acc', 'cape_ml', 'dbz_cmax', ...
os.makedirs(output_dir, exist_ok=True)

ignore_codes = {4}

# ------------------------------
# WW-Farben
# ------------------------------
ww_colors_base = {
    0: "#FFFFFF", 1: "#D3D3D3", 2: "#A9A9A9", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#00FF00", 53: "#00C300", 55: "#009700",
    61: "#00FF00", 63: "#00C300", 65: "#009700",
    80: "#00FF00", 81: "#00C300", 82: "#009700",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    85: "#ADD8E6", 86: "#6495ED",
    77: "#ADD8E6",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}
ww_categories = {
    "Bewölkung": [0, 1, 2, 3],
    "Nebel": [48, 45],
    "Schneeregen": [56, 57],
    "Regen": [61, 63, 65],
    "gefr. Regen": [66, 67],
    "Schnee": [71, 73, 75],
    "Gewitter": [95, 96],
}

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Niederschlags-Farben 1h (tp)
# ------------------------------
prec_bounds = [0.0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4"
])
prec_norm = mcolors.BoundaryNorm(prec_bounds, prec_colors.N)

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.0, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                  125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
tp_acc_norm = mcolors.BoundaryNorm(tp_acc_bounds, tp_acc_colors.N)

# ------------------------------
# CAPE-Farben
# ------------------------------
cape_bounds = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
cape_colors = ListedColormap([
    "#676767", "#006400", "#008000", "#00CC00", "#66FF00", "#FFFF00",
    "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000", "#FF0095",
    "#FC439F", "#FF88D3", "#FF99FF"
])
cape_norm = mcolors.BoundaryNorm(cape_bounds, cape_colors.N)

# ------------------------------
# DBZ-CMAX Farben
# ------------------------------
dbz_bounds = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 67, 70]
dbz_colors = ListedColormap([
    "#676767","#FFFFFF", "#B3EFED", "#8CE7E2", "#00F5ED",
    "#00CEF0", "#01AFF4", "#028DF6", "#014FF7", "#0000F6",
    "#00FF01", "#01DF00", "#00D000", "#00BF00", "#00A701",
    "#019700", "#FFFF00", "#F9F000", "#EDD200", "#E7B500",
    "#FF5000", "#FF2801", "#F40000", "#EA0001", "#CC0000",
    "#FFC8FF", "#E9A1EA", "#D379D3", "#BE55BE", "#960E96"
])
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

# ------------------------------
# Schneehöhen-Farben
# ------------------------------
snow_bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]
snow_colors = ListedColormap([
    "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0582FF",
    "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
    "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
    "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000", "#460000"
])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# ------------------------------
# Schneehöhen-Änderung-Farben
# ------------------------------
change_bounds = [-30, -15, -7, -3, -1, -0.1, 0, 0.1, 0.5, 2, 4, 6, 10, 15, 30, 50, 75, 100]
change_colors = ListedColormap([
    "#D66900", "#F1A30D", "#F7E521", "#78C239", "#24B301",
    "#FFFFFF", "#FFFFFF", "#57A2E3", "#2A78CD", "#124FA8",
    "#2116B0", "#42069C", "#9E009B", "#CB00CC", "#F580F5",
    "#FFB3F4", "#E3001B"
])
change_norm = mcolors.BoundaryNorm(change_bounds, change_colors.N)

# ------------------------------
# Gesamtbewölkung-Farben
# ------------------------------
cloud_bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cloud_colors = ListedColormap([
    "#FFFF00", "#EEEE0B", "#DDDD17", "#CCCC22", "#BBBB2E",
    "#ABAB39", "#9A9A45", "#898950", "#78785C", "#676767"
])
cloud_norm = mcolors.BoundaryNorm(cloud_bounds, cloud_colors.N)

# ------------------------------
# Gesamtwassergehalt
# ------------------------------
twater_bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
twater_colors = ListedColormap([
    "#6E4A00", "#B49E62", "#D7CD13", "#B9F019", "#1ACF06",
    "#08534C", "#035DBE", "#2692FF", "#75BAFF", "#CBBFFF",
    "#EBA6FF", "#DD66FE", "#AC01DD", "#7C009E", "#673775",
    "#6B6B6B", "#818181", "#969696"
])
twater_norm = mcolors.BoundaryNorm(twater_bounds, twater_colors.N)

# ------------------------------
# Schneefallgrenze
# ------------------------------
snowfall_bounds = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
snowfall_colors = ListedColormap([
    "#FF00A6", "#D900FF", "#8C00FF", "#0008FF", "#0099FF",
    "#00F2FF", "#1AFF00", "#FFFB00", "#FFBF00", "#FFA600",
    "#FF6F00", "#930000",
])
snowfall_norm = mcolors.BoundaryNorm(snowfall_bounds, snowfall_colors.N)

# ------------------------------
# Bounding Box ICON-D2 (fix)
# ------------------------------
# Offizielle Domäne des regulären ICON-D2-Gitters des DWD:
# 43.18°N-58.08°N, 3.94°W-20.34°E (0.02°-Gitter, 1215 x 746 Zellen).
extent = [-3.94, 20.34, 43.18, 58.08]  # lon_min, lon_max, lat_min, lat_max

FOOTER_TEXTS = {
    "ww": "Signifikantes Wetter",
    "t2m": "Temperatur 2m (°C)",
    "tp": "Niederschlag, 1Std (mm)",
    "tp_acc": "Akkumulierter Niederschlag (mm)",
    "cape_ml": "CAPE-Index (J/kg)",
    "dbz_cmax": "Sim. max. Radarreflektivität (dBZ)",
    "cloud": "Gesamtbewölkung (%)",
    "wind": "Windböen (km/h)",
    "snow": "Schneehöhe (cm)",
    "change_snow": "Schneehöhenänderung, 6Std (cm)",
    "twater": "Gesamtwassergehalt (mm)",
    "snowfall": "Schneefallgrenze (m)",
}

# Einheit je Variable - für die Wertanzeige im Frontend
VALUE_UNITS = {
    "ww": "",
    "t2m": "°C",
    "tp": "mm",
    "tp_acc": "mm",
    "cape_ml": "J/kg",
    "dbz_cmax": "dBZ",
    "cloud": "%",
    "wind": "km/h",
    "snow": "cm",
    "change_snow": "cm",
    "twater": "mm",
    "snowfall": "m",
}

# Nachkommastellen je Variable für die Wertanzeige
VALUE_DECIMALS = {
    "ww": 0,
    "t2m": 1,
    "tp": 1,
    "tp_acc": 1,
    "cape_ml": 0,
    "dbz_cmax": 0,
    "cloud": 0,
    "wind": 0,
    "snow": 1,
    "change_snow": 1,
    "twater": 1,
    "snowfall": 0,
}

# Sentinel-Wert für "kein Datum/außerhalb" (nicht mehr für Binärdatei
# benötigt, aber zur Referenz im Manifest praktisch)
VALUE_NODATA = -9999.0

# ------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
# ------------------------------
# Leaflet/OSM rendern intern in Web Mercator (EPSG:3857). Unsere GRIB-Daten
# liegen als Plattkarte (EPSG:4326, gleichmäßiges lon/lat-Raster) vor. Ein
# L.imageOverlay dehnt ein rohes EPSG:4326-Bild einfach linear in die
# angegebenen Lat/Lon-Bounds - das ist bei Deutschlands Breitengraden
# (47-56°N) sichtbar falsch (Nord-Süd-Stauchung/Streckung, ca. +20-30%
# Unterschied zwischen Süd- und Nordrand). Daher wird das Datenfeld hier
# vor dem Speichern explizit nach EPSG:3857 umprojiziert, sodass es 1:1
# in die (weiterhin in Lat/Lon angegebenen) Overlay-Bounds passt.
EARTH_RADIUS = 6378137.0  # Meter, WGS84/Web-Mercator-Kugelradius
WEBMERCATOR_WIDTH = 1024   # Ziel-Bildbreite in Pixeln für die Reprojektion


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)  # aufsteigend: Süd -> Nord
    return x_new, y_new


def warp_equirect_to_webmercator(data, lon, lat, extent, method="linear",
                                  out_width=WEBMERCATOR_WIDTH):
    """data/lon/lat: reguläres EPSG:4326-Gitter, lon und lat aufsteigend
    sortiert. Gibt das Datenfeld auf einem regulären EPSG:3857-Pixelraster
    zurück (ebenfalls Süd -> Nord aufsteigend)."""
    x_new, y_new = webmercator_target_grid(extent, out_width=out_width)
    xx, yy = np.meshgrid(x_new, y_new)
    lon_grid = np.degrees(xx / EARTH_RADIUS)
    lat_grid = np.degrees(2 * np.arctan(np.exp(yy / EARTH_RADIUS)) - np.pi / 2)

    interp_func = RegularGridInterpolator(
        (lat, lon), data,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
    pts = np.array([lat_grid.ravel(), lon_grid.ravel()]).T
    warped = interp_func(pts).reshape(lat_grid.shape)
    return warped


# Feste Kartendomäne (ICON-D2-Box) einmalig nach EPSG:3857 (Meter)
# umgerechnet - das ist der "imageExtent", den OpenLayers' ImageStatic-
# Quelle direkt in Kartenkoordinaten erwartet (kein weiteres Strecken
# nötig, da das Bild bereits auf genau dieses Raster projiziert wurde).
_dom_x_min, _dom_y_min = lonlat_to_webmercator(extent[0], extent[2])
_dom_x_max, _dom_y_max = lonlat_to_webmercator(extent[1], extent[3])
DOMAIN_EXTENT_3857 = [float(_dom_x_min), float(_dom_y_min), float(_dom_x_max), float(_dom_y_max)]


def data_to_rgba(data, cmap, norm):
    """Wandelt ein 2D-Datenarray in ein RGBA-uint8-Array um.
    NaN-Werte werden komplett transparent."""
    rgba = cmap(norm(data))  # float RGBA in [0,1], shape (H,W,4)
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba


def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")

    # Verlustfrei speichern: die Colormaps arbeiten mit diskreten Stufen
    # und set_under(alpha=0) für Transparenz - eine verlustbehaftete
    # WebP-Kompression würde Farbgrenzen und den Transparenz-Threshold
    # sichtbar verwischen.
    #
    # method=4 statt 6: bei diesen Bildern (große einfarbige/transparente
    # Flächen, wenige diskrete Farbstufen) liefert method=6 praktisch
    # dieselbe Dateigröße wie method=4, braucht dabei aber ~100x länger
    # (gemessen: 3.9s vs. 0.04s pro Bild) - der höhere Aufwand bringt hier
    # also keinen Vorteil, kostet aber massiv Laufzeit.
    img.save(out_path, format="WEBP", lossless=True, method=4)


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    valid_time_utc_override = None
    lon2d = lat2d = None  # nur gesetzt, falls im Zweig gebraucht

    # Daten je Typ (Logik unverändert aus dem Original übernommen)
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
    else:
        print(f"Unbekannter var_type {var_type}")
        ds.close()
        continue

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

    # ---------------------------------
    # Natives ICON-D2-Gitter beibehalten (keine Interpolation auf ein
    # feineres/anderes Raster mehr) - lediglich sicherstellen, dass lat/lon
    # aufsteigend sortiert sind, da RegularGridInterpolator (in
    # warp_equirect_to_webmercator) das voraussetzt.
    # ---------------------------------
    if lon.ndim == 1 and lat.ndim == 1 and data.ndim == 2:
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[::-1, :]
        if lon[0] > lon[-1]:
            lon = lon[::-1]
            data = data[:, ::-1]

    # data ist jetzt (lat aufsteigend, lon aufsteigend) sortiert,
    # Zeile 0 = Süden. Für save_transparent_webp reicht das - dort wird
    # zum Speichern vertikal gespiegelt (Bildzeile 0 = Norden).

    # ------------------------------
    # Farb-/Kategorie-Mapping je Typ
    # ------------------------------
    if var_type == "t2m":
        cmap, norm = t2m_colors, t2m_norm
        render_data = data
    elif var_type == "ww":
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base and c not in ignore_codes]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes]) if codes else ListedColormap(["#FFFFFF00"])
        norm = mcolors.Normalize(vmin=-0.5, vmax=max(len(codes) - 0.5, 0.5))
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data, fill_value=np.nan, dtype=float)
        for c, i in code2idx.items():
            idx_data[data == c] = i
        render_data = idx_data
    elif var_type == "tp":
        cmap, norm = prec_colors, prec_norm
        render_data = data
    elif var_type == "tp_acc":
        cmap, norm = tp_acc_colors, tp_acc_norm
        render_data = data
    elif var_type == "cape_ml":
        cmap, norm = cape_colors, cape_norm
        render_data = data
    elif var_type == "dbz_cmax":
        render_data = gaussian_filter(data, sigma=0.8)
        cmap, norm = dbz_colors, dbz_norm
    elif var_type == "wind":
        cmap, norm = wind_colors, wind_norm
        render_data = data
    elif var_type == "snow":
        cmap, norm = snow_colors, snow_norm
        render_data = data
    elif var_type == "change_snow":
        cmap, norm = change_colors, change_norm
        render_data = data
    elif var_type == "cloud":
        cmap, norm = cloud_colors, cloud_norm
        render_data = data
    elif var_type == "twater":
        cmap, norm = twater_colors, twater_norm
        render_data = data
    elif var_type == "snowfall":
        cmap, norm = snowfall_colors, snowfall_norm
        render_data = data
    else:
        ds.close()
        continue

    # ------------------------------
    # Nach EPSG:3857 (Web Mercator) umprojizieren
    # ------------------------------
    merc_method = "nearest" if var_type == "ww" else "linear"
    render_data_merc = warp_equirect_to_webmercator(
        render_data, lon, lat, extent, method=merc_method
    )

    # ------------------------------
    # Transparentes WebP speichern
    # ------------------------------
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.webp"
    out_path = os.path.join(output_dir, outname)
    save_transparent_webp(render_data_merc, cmap, norm, out_path)

    print(f"{filename} -> {outname}")

    # ------------------------------
    # Aufräumen - wichtig bei vielen Dateien in der Schleife!
    # ------------------------------
    # cfgrib/xarray hält sonst Dateihandles + gepufferte Arrays offen und
    # der Speicherverbrauch wächst über die Schleife hinweg immer weiter
    # an, bis irgendwann selbst kleine Allokationen fehlschlagen.
    ds.close()
    del data, render_data, render_data_merc
    gc.collect()
