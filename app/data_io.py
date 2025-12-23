from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


ENCODING = "latin1"


def load_wind(path: Path, year: int | None = None) -> pd.DataFrame:
    """Load and clean wind data; optionally prefilter by year (UTC) for speed."""
    df = pd.read_csv(path, sep=";", encoding=ENCODING)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"F": "wind_speed_ms", "D": "wind_dir_deg"})
    wind_speed_rounded = df["wind_speed_ms"].round(1)
    if not np.allclose(df["wind_speed_ms"], wind_speed_rounded, equal_nan=True):
        df["wind_speed_ms"] = wind_speed_rounded
    df["timestamp"] = pd.to_datetime(
        df["MESS_DATUM"].astype(str), format="%Y%m%d%H", errors="coerce", utc=True
    )
    if year is not None:
        start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
        df = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    return df


def load_cloud(path: Path, year: int | None = None) -> pd.DataFrame:
    """Load and clean cloudiness data; optionally prefilter by year (UTC) for speed."""
    df = pd.read_csv(path, sep=";", encoding=ENCODING)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "V_N": "cloud_cover_oktas",
            "V_N_I": "cloud_cover_flag",
            "QN_8": "cloud_qn",
        }
    )
    df["timestamp"] = pd.to_datetime(
        df["MESS_DATUM"].astype(str), format="%Y%m%d%H", errors="coerce", utc=True
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    if year is not None:
        start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
        df = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    return df


def merge_wind_cloud(wind: pd.DataFrame, cloud: pd.DataFrame) -> pd.DataFrame:
    wind = wind.copy()
    cloud = cloud.copy()
    wind["timestamp"] = pd.to_datetime(wind["timestamp"], errors="coerce", utc=True)
    cloud["timestamp"] = pd.to_datetime(cloud["timestamp"], errors="coerce", utc=True)
    merged = wind.merge(
        cloud[
            [
                "STATIONS_ID",
                "timestamp",
                "cloud_qn",
                "cloud_cover_oktas",
                "cloud_cover_flag",
            ]
        ],
        on=["STATIONS_ID", "timestamp"],
        how="left",
    )
    return merged


def filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df.loc[mask].reset_index(drop=True)


def load_and_merge(wind_path: Path, cloud_path: Path, year: int) -> Tuple[pd.DataFrame, Path]:
    wind = load_wind(wind_path, year=year)
    cloud = load_cloud(cloud_path, year=year)
    merged = merge_wind_cloud(wind, cloud)
    merged_year = filter_year(merged, year)
    return merged_year, wind_path.parent
