from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Mapping tables ported from notebook 03
WIND_BINS = [-np.inf, 1.2, 2.3, 3.3, 4.3, np.inf]
WIND_LABELS = ["<=1.2", "1.3-2.3", "2.4-3.3", "3.4-4.3", ">=4.4"]

DAY_CLOUD_BINS = [-np.inf, 2, 5, 8]
DAY_CLOUD_LABELS = ["0-2", "3-5", "6-8"]

NIGHT_CLOUD_BINS = [-np.inf, 6, 8]
NIGHT_CLOUD_LABELS = ["0-6", "7-8"]


MAPPING_DAY: Dict[Tuple[str, str], str] = {
    ("<=1.2", "0-2"): "IV",
    ("<=1.2", "3-5"): "IV",
    ("<=1.2", "6-8"): "IV",
    ("1.3-2.3", "0-2"): "IV",
    ("1.3-2.3", "3-5"): "IV",
    ("1.3-2.3", "6-8"): "III/2",
    ("2.4-3.3", "0-2"): "IV",
    ("2.4-3.3", "3-5"): "IV",
    ("2.4-3.3", "6-8"): "III/2",
    ("3.4-4.3", "0-2"): "IV",
    ("3.4-4.3", "3-5"): "III/2",
    ("3.4-4.3", "6-8"): "III/2",
    (">=4.4", "0-2"): "III/2",
    (">=4.4", "3-5"): "III/1",
    (">=4.4", "6-8"): "III/1",
}

MAPPING_NIGHT: Dict[Tuple[str, str], str] = {
    ("<=1.2", "0-6"): "I",
    ("<=1.2", "7-8"): "II",
    ("1.3-2.3", "0-6"): "I",
    ("1.3-2.3", "7-8"): "II",
    ("2.4-3.3", "0-6"): "II",
    ("2.4-3.3", "7-8"): "III/1",
    ("3.4-4.3", "0-6"): "III/1",
    ("3.4-4.3", "7-8"): "III/1",
    (">=4.4", "0-6"): "III/1",
    (">=4.4", "7-8"): "III/1",
}

# Transition table around sunrise/sunset; value = (default, alternative, rule)
TRANSITION_MAP: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {
    ("I", "IV", "SA+1..SA+2"): ("I", "II", "a"),
    ("I", "IV", "SA+2..SA+3"): ("II", None, None),
    ("I", "IV", "SU-2..SU-1"): ("II", None, None),
    ("I", "IV", "SU-1..SU"): ("II", "I", "b"),
    ("I", "IV", "SU..SU+1"): ("I", "II", "a"),
    ("I", "III/2", "SA+1..SA+2"): ("II", None, None),
    ("I", "III/2", "SA+2..SA+3"): ("II", None, None),
    ("I", "III/2", "SU-2..SU-1"): ("III/1", None, None),
    ("I", "III/2", "SU-1..SU"): ("III/1", None, None),
    ("I", "III/2", "SU..SU+1"): ("I", "II", "a"),
    ("II", "IV", "SA+1..SA+2"): ("II", None, None),
    ("II", "IV", "SA+2..SA+3"): ("III/1", None, None),
    ("II", "IV", "SU-2..SU-1"): ("III/1", None, None),
    ("II", "IV", "SU-1..SU"): ("II", None, None),
    ("II", "IV", "SU..SU+1"): ("II", None, None),
    ("II", "III/2", "SA+1..SA+2"): ("III/1", None, None),
    ("II", "III/2", "SA+2..SA+3"): ("III/1", None, None),
    ("II", "III/2", "SU-2..SU-1"): ("III/1", None, None),
    ("II", "III/2", "SU-1..SU"): ("III/1", None, None),
    ("II", "III/2", "SU..SU+1"): ("II", None, None),
    ("III/1", "IV", "SA+1..SA+2"): ("III/1", None, None),
    ("III/1", "IV", "SA+2..SA+3"): ("III/2", None, None),
    ("III/1", "IV", "SU-2..SU-1"): ("III/2", None, None),
    ("III/1", "IV", "SU-1..SU"): ("III/1", None, None),
    ("III/1", "IV", "SU..SU+1"): ("III/1", None, None),
    ("III/1", "III/2", "SA+1..SA+2"): ("III/1", None, None),
    ("III/1", "III/2", "SA+2..SA+3"): ("III/1", None, None),
    ("III/1", "III/2", "SU-2..SU-1"): ("III/2", None, None),
    ("III/1", "III/2", "SU-1..SU"): ("III/2", None, None),
    ("III/1", "III/2", "SU..SU+1"): ("III/1", None, None),
    ("III/1", "III/1", "SA+1..SA+2"): ("III/1", None, None),
    ("III/1", "III/1", "SA+2..SA+3"): ("III/1", None, None),
    ("III/1", "III/1", "SU-2..SU-1"): ("III/1", None, None),
    ("III/1", "III/1", "SU-1..SU"): ("III/1", None, None),
    ("III/1", "III/1", "SU..SU+1"): ("III/1", None, None),
}


def _resolve_transition(kn, kt, win, base, month, wind, cloud):
    val = TRANSITION_MAP.get((kn, kt, win))
    if val is None:
        return base
    default, alt, rule = val
    if alt and rule == "a":
        if (3 <= month <= 11) and (wind >= 1.3):
            return alt
    if alt and rule == "b":
        if (month in (1, 2, 12)) and (wind < 1.3) and (cloud <= 6):
            return alt
    return default


def compute_classes(df: pd.DataFrame, tz_name: str = "UTC") -> pd.DataFrame:
    df = df.copy()
    df["timestamp_local"] = pd.to_datetime(df.get("timestamp_local", df["timestamp"]), errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    invalid_wind = (df["wind_speed_ms"] == -999) | (df["wind_dir_deg"] == -999)
    wind_speed_clean = df["wind_speed_ms"].where(~invalid_wind)

    wind_bin = pd.cut(wind_speed_clean, bins=WIND_BINS, labels=WIND_LABELS)
    day_cloud_bin = pd.cut(df["cloud_cover_oktas"], bins=DAY_CLOUD_BINS, labels=DAY_CLOUD_LABELS)
    night_cloud_bin = pd.cut(df["cloud_cover_oktas"], bins=NIGHT_CLOUD_BINS, labels=NIGHT_CLOUD_LABELS)

    df["klasse_kt"] = [MAPPING_DAY.get(k) for k in zip(wind_bin.astype(object), day_cloud_bin.astype(object))]
    df["klasse_kn"] = [MAPPING_NIGHT.get(k) for k in zip(wind_bin.astype(object), night_cloud_bin.astype(object))]
    df["ausbreitungsklasse_base"] = np.where(df["day_night"] == True, df["klasse_kt"], df["klasse_kn"])

    ts_local = pd.to_datetime(df["timestamp_local"], errors="coerce")
    sa_utc = pd.to_datetime(df["SA"], errors="coerce", utc=True)
    su_utc = pd.to_datetime(df["SU"], errors="coerce", utc=True)
    delta_sa = pd.TimedeltaIndex(ts_local - sa_utc).total_seconds() / 3600
    delta_su = pd.TimedeltaIndex(ts_local - su_utc).total_seconds() / 3600

    window = pd.Series(index=df.index, dtype="object")
    window.loc[(delta_sa >= 0) & (delta_sa < 1)] = "SA+1..SA+2"
    window.loc[(delta_sa >= 1) & (delta_sa < 2)] = "SA+1..SA+2"
    window.loc[(delta_sa >= 2) & (delta_sa < 3)] = "SA+2..SA+3"
    window.loc[(delta_su >= -2) & (delta_su < -1)] = "SU-2..SU-1"
    window.loc[(delta_su >= -1) & (delta_su < 0)] = "SU-1..SU"
    window.loc[(delta_su >= 0) & (delta_su < 1)] = "SU..SU+1"

    months = pd.DatetimeIndex(df["timestamp_local"]).month
    wind_for_logic = wind_speed_clean
    cloud_for_logic = df["cloud_cover_oktas"]
    df["ausbreitungsklasse"] = [
        _resolve_transition(kn, kt, win, base, m, w, c)
        for kn, kt, win, base, m, w, c in zip(
            df["klasse_kn"],
            df["klasse_kt"],
            window,
            df["ausbreitungsklasse_base"],
            months,
            wind_for_logic,
            cloud_for_logic,
        )
    ]

    cloud_missing = df["cloud_cover_oktas"].isna()
    wind_bins_nc = [-np.inf, 2.3, 3.3, np.inf]
    wind_labels_nc = ["<=2.3", "2.4-3.3", ">=3.4"]
    wind_bin_nc = pd.cut(wind_for_logic, bins=wind_bins_nc, labels=wind_labels_nc)

    window_nc = pd.Series(index=df.index, dtype="object")
    window_nc.loc[(delta_sa < 1) | (delta_su >= 0)] = "SU..SA+1"
    window_nc.loc[(delta_sa >= 1) & (delta_sa < 3)] = "SA+1..SA+3"
    window_nc.loc[(delta_sa >= 3) & (delta_su <= -2)] = "SA+3..SU-2"
    window_nc.loc[(delta_su > -2) & (delta_su < 0)] = "SU-2..SU"

    mapping_nc = {
        ("<=2.3", "SU..SA+1"): "I",
        ("<=2.3", "SA+1..SA+3"): "II",
        ("<=2.3", "SA+3..SU-2"): "III/2",
        ("<=2.3", "SU-2..SU"): "III/1",
        ("2.4-3.3", "SU..SA+1"): "II",
        ("2.4-3.3", "SA+1..SA+3"): "III/1",
        ("2.4-3.3", "SA+3..SU-2"): "III/2",
        ("2.4-3.3", "SU-2..SU"): "III/1",
        (">=3.4", "SU..SA+1"): "III/1",
        (">=3.4", "SA+1..SA+3"): "III/1",
        (">=3.4", "SA+3..SU-2"): "III/1",
        (">=3.4", "SU-2..SU"): "III/1",
    }
    df["ausbreitungsklasse_no_cloud"] = [mapping_nc.get(k) for k in zip(wind_bin_nc.astype(object), window_nc)]
    df["ausbreitungsklasse"] = np.where(
        cloud_missing, df["ausbreitungsklasse_no_cloud"], df["ausbreitungsklasse"]
    )

    class_order = ["I", "II", "III/1", "III/2", "IV", "V"]
    _next_class = {c: class_order[min(i + 1, len(class_order) - 1)] for i, c in enumerate(class_order)}

    def _upgrade(series, mask):
        mapped = series.map(_next_class).fillna(series)
        return pd.Series(np.where(mask, mapped, series), index=series.index)

    hours_local = pd.DatetimeIndex(df["timestamp_local"]).hour
    summer = months.isin([6, 7, 8])
    mask_summer1 = summer & (hours_local >= 10) & (hours_local <= 16) & (
        (df["cloud_cover_oktas"] <= 6) | ((df["cloud_cover_oktas"] == 7) & (wind_for_logic < 2.4))
    )
    mask_summer2 = summer & (hours_local >= 12) & (hours_local <= 15) & (df["cloud_cover_oktas"] <= 5)
    mask_may_sep = months.isin([5, 9]) & (hours_local >= 11) & (hours_local <= 15) & (df["cloud_cover_oktas"] <= 6)

    df["ausbreitungsklasse"] = _upgrade(df["ausbreitungsklasse"], mask_summer1)
    df["ausbreitungsklasse"] = _upgrade(df["ausbreitungsklasse"], mask_summer2)
    df["ausbreitungsklasse"] = _upgrade(df["ausbreitungsklasse"], mask_may_sep)

    winter_mask = months.isin([12, 1, 2])
    df.loc[winter_mask & (df["ausbreitungsklasse"] == "IV"), "ausbreitungsklasse"] = "III/2"

    null_class = invalid_wind
    df.loc[null_class, "ausbreitungsklasse"] = np.nan
    return df


def export_akterm(df: pd.DataFrame, year: int, out_path: Path, tz_name: str = "UTC") -> Path:
    ts_local = pd.to_datetime(df["timestamp_local"], errors="coerce")
    ts_idx = pd.DatetimeIndex(ts_local)
    if ts_idx.tz is None:
        ts_idx = ts_idx.tz_localize(tz_name)
    ts_utc = ts_idx.tz_convert("UTC")
    ts_out = pd.DatetimeIndex(ts_utc).tz_localize(None)

    mask_year = (ts_out >= pd.Timestamp(f"{year}-01-01 00:00:00")) & (
        ts_out < pd.Timestamp(f"{year + 1}-01-01 00:00:00")
    )
    ts_out = ts_out[mask_year]
    df_masked = df.loc[mask_year].reset_index(drop=True)

    lines = []
    kennzahl_map = {"I": "1", "II": "2", "III/1": "3", "III/2": "4", "IV": "5", "V": "6"}
    for ts, sid, wdir, ws, ak in zip(
        ts_out, df_masked["STATIONS_ID"], df_masked["wind_dir_deg"], df_masked["wind_speed_ms"], df_masked["ausbreitungsklasse"]
    ):
        sid_str = "" if pd.isna(sid) else f"{int(sid):05d}"
        wdir_str = "" if pd.isna(wdir) else f"{int(round(wdir))}"
        ws_str = "" if pd.isna(ws) else f"{int(round(ws * 10))}"
        if pd.isna(ak):
            ak_str = ""
            main_ak = ""
            fehlwert = "-999"
        else:
            ak_str = ak
            main_ak = kennzahl_map.get(str(ak), "")
            fehlwert = ""

        if pd.isna(ts):
            year_str = month = day = hour = minute = ""
        else:
            year_str = f"{ts.year:04d}"
            month = f"{ts.month:02d}"
            day = f"{ts.day:02d}"
            hour = f"{ts.hour:02d}"
            minute = f"{ts.minute:02d}"

        lines.append(
            f"AK {sid_str} {year_str} {month} {day} {hour} {minute} {wdir_str} {ws_str} {ak_str} {main_ak} {fehlwert}"
        )

    header_lines = [
        "* AKTERM-Zeitreihe, Bayerisches Landesamt fuer Umwelt (LfU)",
        "* Station Muenchen-Stadt mit Bedeckung Muenchen-Stadt",
        f"* Zeitraum 01.01.{year} - 31.12.{year} (UTC).",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        for line in lines:
            f.write(line + "\n")
    return out_path
