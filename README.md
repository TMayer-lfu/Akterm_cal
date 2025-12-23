# Akterm_analysis

## Deployment (Streamlit Cloud / Render)

### Streamlit Community Cloud
1. Create a new app from this repository.
2. Set the main file to `app/ui/streamlit_app.py`.
3. Add a Python version file (`runtime.txt`) if you want to pin a version (e.g. `python-3.11`); the app is tested with Python 3.11.
4. Use the logs panel to check whether dependencies finished installing. The app should start once provisioning is done.

### Render
1. Use the included `render.yaml` to create a new Web Service.
2. The `startCommand` already binds Streamlit to `0.0.0.0` and uses the Render-provided `$PORT`, which avoids the “Provisioning machine…” stall caused by a service that never passes the health check.
3. If the service still hangs on provisioning, inspect the build logs for pip failures or missing network access to DWD data paths. Render restarts automatically when the health check fails; viewing live logs usually exposes the root cause.

### Local
```
pip install -r requirements.txt
PYTHONPATH=. streamlit run app/ui/streamlit_app.py
```