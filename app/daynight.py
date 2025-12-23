from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun
import pandas as pd


def sunrise_sunset(lat: float, lon: float, d: date, tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)
    s = sun(loc.observer, date=d, tzinfo=tz)
    return s["sunrise"], s["sunset"]


def _dec_hours(dt: datetime) -> float:
    return dt.hour + dt.minute / 60 + dt.second / 3600 + dt.microsecond / 3.6e9


def classify_day_night_vdi(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    lat: float = 48.1372,
    lon: float = 11.5756,
    tz_name: str = "UTC",
) -> pd.DataFrame:
    """VDI-like rule: night if SU < t <= SA+1 (local decimal hours); else day."""
    tz = ZoneInfo(tz_name)
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["ts_local"] = pd.DatetimeIndex(df[ts_col]).tz_convert(tz)
    if df[ts_col].isna().any():
        raise ValueError(f"{ts_col} contains non-datetime values")

    dates = sorted(pd.unique(pd.DatetimeIndex(df["ts_local"]).date))
    sun_map: Dict[date, Tuple[datetime, datetime]] = {
        d: sunrise_sunset(lat, lon, d, tz_name) for d in dates
    }

    def flag(row):
        dt_local = row["ts_local"]
        d = dt_local.date()
        sr, ss = sun_map[d]
        h = _dec_hours(dt_local)
        sr_h = _dec_hours(sr.astimezone(tz))
        ss_h = _dec_hours(ss.astimezone(tz))
        return (h > ss_h) or (h <= sr_h + 1)

    df["SA"] = [sun_map[d][0] for d in pd.DatetimeIndex(df["ts_local"]).date]
    df["SU"] = [sun_map[d][1] for d in pd.DatetimeIndex(df["ts_local"]).date]
    df["day_night"] = ~df.apply(flag, axis=1)  # True = day
    return df
