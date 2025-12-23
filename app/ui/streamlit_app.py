from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from app import pipeline, config


def _persist_upload(upload, suffix: str) -> Optional[Path]:
    if upload is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    return Path(tmp.name)


def _resolve_upload_or_path(upload, fallback: str, suffix: str = ".txt") -> Path:
    """Return a Path from upload (stored temp) or fallback path string."""
    path = _persist_upload(upload, suffix=suffix) if upload else None
    return path if path is not None else Path(fallback)


def main():
    st.set_page_config(page_title="WINMISKAM Pipeline", layout="wide")
    st.title("WINMISKAM Pipeline (Notebooks 01–03)")

    st.sidebar.header("Eingaben")
    wind_upload = st.sidebar.file_uploader(
        "Wind-Datei (03379_*stadt.txt)", type=["txt", "csv"]
    )
    cloud_upload = st.sidebar.file_uploader(
        "Cloud-Datei (produkt_n_stunde_*.txt)", type=["txt", "csv"]
    )
    year = st.sidebar.selectbox("Jahr", list(range(2000, 2026)), index=9)
    lat = st.sidebar.number_input(
        "Breite (lat)", value=config.DEFAULT_LAT, format="%0.4f"
    )
    lon = st.sidebar.number_input(
        "Länge (lon)", value=config.DEFAULT_LON, format="%0.4f"
    )
    tz = st.sidebar.text_input("Zeitzone", value=config.DEFAULT_TZ)
    out_dir = st.sidebar.text_input("Output-Ordner", value="data/processed")
    project = st.sidebar.text_input("Projektname", value="muenchen_stadt")

    st.sidebar.write("Alternativ: feste Pfade eingeben, wenn Uploads leer bleiben.")
    wind_path_txt = st.sidebar.text_input(
        "Wind-Pfad", value="data/raw/wind/03379_Muenchen_stadt.txt"
    )
    cloud_path_txt = st.sidebar.text_input(
        "Cloud-Pfad",
        value="data/raw/cloudiness/produkt_n_stunde_19790101_20241231_03379.txt",
    )

    st.sidebar.write("Assumptions werden unten angezeigt und als JSON gespeichert.")

    status = st.empty()
    if st.sidebar.button("Pipeline starten"):
        wind_path = _resolve_upload_or_path(wind_upload, wind_path_txt)
        cloud_path = _resolve_upload_or_path(cloud_upload, cloud_path_txt)
        cfg = config.PipelineConfig(
            year=year,
            wind_path=wind_path,
            cloud_path=cloud_path,
            out_dir=Path(out_dir),
            project_name=project,
            lat=lat,
            lon=lon,
            tz_name=tz,
        )
        try:
            status.info("Pipeline läuft...")
            paths = pipeline.run_pipeline(cfg)
            status.success("Fertig.")
            st.success(f"Ergebnisse geschrieben nach {paths['ausbreitung'].parent}")

            with open(paths["ausbreitung"], "rb") as f:
                st.download_button(
                    "Ausbreitung CSV laden", data=f, file_name=paths["ausbreitung"].name
                )
            with open(paths["akterm"], "rb") as f:
                st.download_button(
                    "AKTERM laden", data=f, file_name=paths["akterm"].name
                )
            with open(paths["assumptions"], "rb") as f:
                st.download_button(
                    "assumptions.json laden", data=f, file_name="assumptions.json"
                )
        except Exception as exc:  # pragma: no cover - UI feedback
            status.error(f"Fehler: {exc}")

    st.header("Annahmen")
    cfg_preview = config.PipelineConfig(
        year=year,
        wind_path=Path(wind_path_txt),
        cloud_path=Path(cloud_path_txt),
        out_dir=Path(out_dir),
        project_name=project,
        lat=lat,
        lon=lon,
        tz_name=tz,
    )
    st.json(config.assumptions(cfg_preview))


if __name__ == "__main__":
    main()
