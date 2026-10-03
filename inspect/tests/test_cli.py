"""Install the package into a scratch venv and run the inspect CLI (mockllm only)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import STAGED_ROOT

PKG = Path(__file__).resolve().parents[1]
UV = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


@pytest.mark.slow
@pytest.mark.skipif(not Path(UV).exists(), reason="uv not found")
def test_inspect_cli_smoke(tmp_path, data_root):
    venv = tmp_path / "venv"
    subprocess.run([UV, "venv", "-q", str(venv)], check=True)
    py = venv / "bin" / "python"
    subprocess.run([UV, "pip", "install", "-q", "--python", str(py), str(PKG)], check=True)

    # real staged data when available, otherwise the fixture package
    data = STAGED_ROOT if (STAGED_ROOT / "steer_me").exists() else data_root
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["STEER_BENCH_DATA_DIR"] = str(data)
    cmd = [str(venv / "bin" / "inspect"), "eval", "steer_bench/steer_me", "-T", "element=consumer_surplus",
           "--model", "mockllm/model", "--limit", "5", "--log-dir", str(tmp_path / "logs"), "--display", "plain"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=tmp_path, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    logs = list((tmp_path / "logs").glob("*.eval"))
    assert len(logs) == 1

    check = (
        "import sys; from inspect_ai.log import read_eval_log; "
        "log = read_eval_log(sys.argv[1]); "
        "assert log.status == 'success', log.status; "
        "assert len(log.samples) == 5; "
        "assert log.eval.task == 'steer_bench/steer_me', log.eval.task; "
        "assert all(s.metadata['element'] == 'consumer_surplus' for s in log.samples); "
        "print('ok', log.eval.task, len(log.samples))"
    )
    r = subprocess.run([str(py), "-c", check, str(logs[0])], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0 and r.stdout.startswith("ok"), r.stdout + r.stderr

    r = subprocess.run([str(venv / "bin" / "steer-bench-export"), str(tmp_path / "logs"), "--out",
                        str(tmp_path / "cells")], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "cells" / "cells.parquet").exists()
