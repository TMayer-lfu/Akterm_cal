from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from . import data_io, daynight, ausbreitung, config


def run_pipeline(cfg: config.PipelineConfig) -> Dict[str, Path]:
    wind_path = Path(cfg.wind_path)
    cloud_path = Path(cfg.cloud_path)
    if not wind_path.exists():
        raise FileNotFoundError(wind_path)
    if not cloud_path.exists():
        raise FileNotFoundError(cloud_path)

    out_root = cfg.resolved_out_dir() / cfg.project_name
    out_root.mkdir(parents=True, exist_ok=True)

    merged, _ = data_io.load_and_merge(wind_path, cloud_path, cfg.year)
    merged = daynight.classify_day_night_vdi(
        merged, ts_col="timestamp", lat=cfg.lat, lon=cfg.lon, tz_name=cfg.tz_name
    )
    merged["timestamp_local"] = merged["timestamp"]  # keep explicit local copy

    merged_day_night_path = out_root / f"merged_wind_cloud_{cfg.year}_day_night.csv"
    merged.to_csv(merged_day_night_path, index=False)

    klass_df = ausbreitung.compute_classes(merged, tz_name=cfg.tz_name)
    ausbreitung_path = out_root / f"merged_wind_cloud_{cfg.year}_day_night_ausbreitung.csv"
    klass_df.to_csv(ausbreitung_path, index=False)

    akterm_path = out_root / f"akterm_{cfg.year}_{cfg.project_name}.akt"
    ausbreitung.export_akterm(klass_df, cfg.year, akterm_path, tz_name=cfg.tz_name)

    assumptions_path = out_root / "assumptions.json"
    with assumptions_path.open("w", encoding="utf-8") as f:
        json.dump(config.assumptions(cfg), f, indent=2)

    return {
        "merged_day_night": merged_day_night_path,
        "ausbreitung": ausbreitung_path,
        "akterm": akterm_path,
        "assumptions": assumptions_path,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WINMISKAM pipeline (notebooks 01-03)")
    p.add_argument("--year", type=int, required=True, help="Target year")
    p.add_argument("--wind", type=Path, required=True, help="Wind file path (e.g. 03379_*stadt.txt)")
    p.add_argument("--cloud", type=Path, required=True, help="Cloudiness file path (produkt_n_stunde_*.txt)")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output directory")
    p.add_argument("--project", type=str, default="muenchen_stadt", help="Project name / subfolder")
    p.add_argument("--lat", type=float, default=config.DEFAULT_LAT, help="Latitude")
    p.add_argument("--lon", type=float, default=config.DEFAULT_LON, help="Longitude")
    p.add_argument("--tz", type=str, default=config.DEFAULT_TZ, help="Timezone name")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config.PipelineConfig(
        year=args.year,
        wind_path=args.wind,
        cloud_path=args.cloud,
        out_dir=args.out_dir,
        project_name=args.project,
        lat=args.lat,
        lon=args.lon,
        tz_name=args.tz,
    )
    paths = run_pipeline(cfg)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
