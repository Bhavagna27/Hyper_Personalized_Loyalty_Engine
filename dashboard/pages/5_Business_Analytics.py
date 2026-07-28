from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.theme import configure_page
from dashboard.core import render_business_analytics_page


configure_page("Business Analytics | Hyper-Personalized Loyalty Engine")
render_business_analytics_page()

