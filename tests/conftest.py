"""Load the project source with the installed runtime dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workloads/vllm/vllm"))
