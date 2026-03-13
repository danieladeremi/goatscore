# Run Locally (Outside Zerve)

This project was authored as a graph of script blocks that share in-memory variables.
Use `run_pipeline.py` to replay that behavior in VS Code/terminal.

## 1) Create and activate a virtual environment

From this folder:

```powershell
cd "C:\Users\danny\OneDrive\Documents\New project\goatscore\f93beda2-65f8-4f28-9d09-2eadd93e2b06\NBA All-Time Greatness Predictor"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py` is unavailable, use your Python executable directly:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Run pipeline

Full graph order:

```powershell
python run_pipeline.py --mode full
```

Data build only (through validation CSV exports):

```powershell
python run_pipeline.py --mode data-only
```

Continue even if one step fails:

```powershell
python run_pipeline.py --mode full --continue-on-error
```

Run explicit scripts in order:

```powershell
python run_pipeline.py --steps "load_historical_players.py,scrape_career_stats.py,build_full_dataset.py"
```

## Notes

- Keep your terminal working directory in this folder so CSV reads/writes resolve correctly.
- The pipeline scrapes Basketball-Reference, so internet access is required.
- Several scripts generate matplotlib outputs with non-interactive backend (`Agg`), so they run headless fine.
