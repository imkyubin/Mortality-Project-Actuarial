"""View 2: state x sex x cause comparison, single-age 1999-2024 data only.

TODO:
- render(): the page function st.navigation calls.
- State, sex and cause filters, then data.lookup_combination for that combination.
- Credibility gate: if the combination was dropped, show its Drop Reason as a message in place
  of a chart, then stop.
- Age axis follows the combination's Resolution (single-year ages or the grouped bins); never
  mix the two.
- If the young end was truncated, say so on the page.
- Age and year filters, then death rate per 100,000 charts.
- Later: the AI Insight panel (ai_reporting), with its required disclaimer.

Reference: templates/dashboard/subsection_view.py
"""
