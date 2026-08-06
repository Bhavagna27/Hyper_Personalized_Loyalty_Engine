import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.safe_html import css_token, esc
from loyalty_engine.config import PATHS
from loyalty_engine.security import is_trusted_artifact_path, load_artifact


def test_esc_neutralizes_html_payloads():
    assert esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert esc('" onmouseover="alert(1)') == "&quot; onmouseover=&quot;alert(1)"
    assert esc(None, "N/A") == "N/A"


def test_css_token_strips_attribute_breakouts():
    assert css_token("ready' onclick='alert(1)") == "readyonclickalert1"
    assert css_token("!!!", default="missing") == "missing"


def test_artifacts_inside_project_are_trusted():
    assert is_trusted_artifact_path(PATHS.kmeans_model_path)


def test_load_artifact_rejects_untrusted_path(tmp_path):
    with pytest.raises(PermissionError):
        load_artifact(tmp_path / "model.joblib")
