from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from loyalty_engine.io.persistence import ensure_dir


def plot_customer_overview(dataset: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=dataset,
        y="Membership_Tier",
        order=dataset["Membership_Tier"].value_counts().index,
        ax=ax,
    )
    ax.set_title("Membership Tier Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "membership_tier_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(dataset["Total_Spend_6M"], kde=True, ax=ax)
    ax.set_title("Total Spend in Last 6 Months")
    fig.tight_layout()
    fig.savefig(output_dir / "total_spend_distribution.png", dpi=150)
    plt.close(fig)

