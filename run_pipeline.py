"""
run_pipeline.py
Runs the full Seattle event-demand pipeline:
  1. Fetch / generate data
  2. Train model and output predictions
  3. Build interactive map
"""
import subprocess
import sys
from pathlib import Path

def run(script: str):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run([sys.executable, script], check=True)
    return result

if __name__ == "__main__":
    base = Path(__file__).parent
    run(str(base / "data" / "fetch_data.py"))
    run(str(base / "src"  / "model.py"))
    run(str(base / "src"  / "visualize.py"))
    print("\nPipeline complete.")
    print("Open the map: open outputs/seattle_demand_map.html")
