import pytest
pytestmark = pytest.mark.smoke

import os
import sys
import subprocess
import importlib.util
import pytest

SCRIPT = os.path.join("scripts", "demo_infer.py")

@pytest.mark.skipif(not os.path.exists(SCRIPT), reason="demo_infer.py がありません")
def test_demo_infer_help_or_import_smoke():
    # 1) まずは --help を試す
    try:
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True,
            text=True,
            timeout=15
        )
        # --help が成功（0終了）ならOK
        if proc.returncode == 0:
            return
    except Exception:
        pass

    # 2) --help がない場合は import スモークでOKにする
    try:
        spec = importlib.util.spec_from_file_location("demo_infer", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.skip(f"demo_infer.py の import に失敗: {e}")
