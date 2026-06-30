"""Load CDC WONDER mortality exports.

Purpose:
    Read raw CDC WONDER CSV files while preserving deaths and population
    exposures for actuarial and statistical modeling.

First implementation target:
    Write a small function that loads one WONDER CSV and removes footer notes.
"""

from io import StringIO
from pathlib import Path

import pandas as pd


def load_wonder_csv(path):
    """Load a CDC WONDER CSV and ignore footer notes."""
    path = Path(path)

    data_lines = []

    # The with statement closes the file automatically after reading.
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            # Stop before CDC WONDER footer notes and citations.
            if line.startswith('"---"') or line.startswith("---"):
                break

            data_lines.append(line)

    return pd.read_csv(StringIO("".join(data_lines)))
