import subprocess
import sys

python_path = sys.executable   # ✅ current env python

# Start FastAPI
api_process = subprocess.Popen(
    [python_path, "-m", "uvicorn", "src.api:app", "--reload"]
)

# Start Streamlit
ui_process = subprocess.Popen(
    [python_path, "-m", "streamlit", "run", "app.py"]
)

api_process.wait()
ui_process.wait()