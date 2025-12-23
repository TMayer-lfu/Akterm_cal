from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


DEFAULT_LAT = 48.1372
DEFAULT_LON = 11.5756
DEFAULT_TZ = "UTC"


@dataclass
class PipelineConfig:
    year: int
    wind_path: Path
    cloud_path: Path
    out_dir: Path
    project_name: str = "muenchen_stadt"
    lat: float = DEFAULT_LAT
    lon: float = DEFAULT_LON
    tz_name: str = DEFAULT_TZ

    def resolved_out_dir(self) -> Path:
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


def assumptions(cfg: PipelineConfig) -> Dict[str, Any]:
    return {
        "timezone_input": "UTC (DWD Timestamps)",
        "timezone_local": cfg.tz_name,
        "location": {"lat": cfg.lat, "lon": cfg.lon},
        "wind": {
            "rounding_ms": 0.1,
            "invalid_marker": -999,
            "direction_col": "wind_dir_deg",
            "speed_col": "wind_speed_ms",
        },
        "cloud": {
            "invalid_marker": -999,
            "bins_day": "0-2 / 3-5 / 6-8",
            "bins_night": "0-6 / 7-8",
            "fallback_no_cloud": True,
        },
        "vdi_thresholds": {
            "wind_bins": [1.2, 2.3, 3.3, 4.3],
            "transition_windows": [
                "SA+1..SA+2",
                "SA+2..SA+3",
                "SU-2..SU-1",
                "SU-1..SU",
                "SU..SU+1",
            ],
            "summer_uplift": True,
            "winter_rule": "IV -> III/2 in Dec/Jan/Feb",
        },
        "year": cfg.year,
        "project": cfg.project_name,
        "outputs": {
            "csv": f"merged_wind_cloud_{cfg.year}_day_night_ausbreitung.csv",
            "akterm": f"akterm_{cfg.year}_{cfg.project_name}.akt",
            "assumptions": "assumptions.json",
        },
    }
