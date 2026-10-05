"""Shared chart helpers for both views.

TODO:
- One line-chart helper (Altair or Plotly) with hover tooltips showing derived values only:
  rates, ex, never Deaths/Population.
- Fixed colors per entity (a color per sex, a color per age band), so changing a filter never
  repaints an existing series a different color.

Reference: templates/dashboard/charts.py
"""
