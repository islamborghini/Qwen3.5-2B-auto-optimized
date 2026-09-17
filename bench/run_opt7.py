import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.run([sys.executable, os.path.join(here, "optimize.py"), "--max_candidates", "4", "--max_minutes", "30", "--reps", "3"]).returncode)
