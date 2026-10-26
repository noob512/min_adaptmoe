"""Use the reviewed source with the compiled dependencies installed by the user."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workloads/vllm/vllm"))
