"""Small, disclosed CI-failure fixture for the coding-agent demonstration."""

from dataclasses import dataclass
from pathlib import Path


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "cart_project"

CODING_QUESTION = {
    "type": "choice",
    "instructions": "Which investigation route best fits this CI failure evidence?",
    "criteria": {
        "code_bug": "A reproducible failure caused by incorrect project source behavior; inspect code and tests.",
        "environment": "A missing dependency, incompatible runtime, or CI configuration problem; inspect environment setup.",
        "test_flake": "An intermittent timeout or race that passes on rerun; inspect test stability and timing.",
        "unknown": "The evidence is too incomplete to choose a cause; request more logs or reproduction details.",
    },
}


@dataclass(frozen=True)
class CodingCase:
    name: str
    log: str
    expected: str
    test_id: str | None = None


CASES = [
    CodingCase(
        "Discount regression",
        "CI pytest: tests/test_cart.py::test_percentage_discount failed reproducibly on 3 reruns. "
        "AssertionError: assert total([Item(price=10, quantity=2)], discount=0.10) == 18.0; got 19.9. "
        "Trace points to cart.py:total.",
        "code_bug",
        "tests/test_cart.py::test_percentage_discount",
    ),
    CodingCase(
        "Shipping boundary",
        "CI pytest: tests/test_cart.py::test_free_shipping_at_threshold failed on 3 reruns. "
        "AssertionError: assert shipping_fee(50.0) == 0.0; got 5.0. Trace points to cart.py:shipping_fee.",
        "code_bug",
        "tests/test_cart.py::test_free_shipping_at_threshold",
    ),
    CodingCase(
        "Zero quantity",
        "CI pytest: tests/test_cart.py::test_zero_quantity_rejected failed reproducibly. "
        "AssertionError: did not raise ValueError for Item(price=5, quantity=0). Trace points to cart.py:validate_item.",
        "code_bug",
        "tests/test_cart.py::test_zero_quantity_rejected",
    ),
    CodingCase(
        "Missing dependency",
        "CI setup failed before tests ran: ModuleNotFoundError: No module named 'pytest'. "
        "The runner installed production dependencies only; pytest is absent from the test environment.",
        "environment",
    ),
    CodingCase(
        "Wrong Python version",
        "CI setup failed before tests ran: package requires Python >=3.12, runner uses Python 3.10. "
        "No application assertion was executed.",
        "environment",
    ),
    CodingCase(
        "Transient timeout",
        "CI pytest: test_checkout_api timed out once after 2 seconds while waiting for a mock server. "
        "The same commit passed on two immediate reruns with no code changes.",
        "test_flake",
    ),
    CodingCase(
        "Rerun passed",
        "CI pytest: test_refresh_token failed with an intermittent timing race. "
        "A retry on the same commit passed; failure appears only under parallel test load.",
        "test_flake",
    ),
    CodingCase(
        "Truncated log",
        "CI status: failed. The log upload was truncated before the failing command, stack trace, or test name. "
        "No reproduction details are available.",
        "unknown",
    ),
]
