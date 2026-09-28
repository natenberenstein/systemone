import subprocess
import sys

from demo.coding_data import FIXTURE_ROOT


def test_cart_fixture_has_three_real_failures_and_one_pass():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_cart.py", "-q"],
        cwd=FIXTURE_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 1
    assert "3 failed, 1 passed" in completed.stdout
    assert "19.9 == 18.0" in completed.stdout
    assert "5.0 == 0.0" in completed.stdout
    assert "DID NOT RAISE ValueError" in completed.stdout
