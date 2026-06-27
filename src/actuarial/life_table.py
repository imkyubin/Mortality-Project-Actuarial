"""Life table construction.

Purpose:
    Convert age-specific mortality rates into qx, px, lx, dx, Lx, Tx, and ex.

First implementation target:
    Build one pure function that accepts a cleaned single-year age table for
    one year and one sex, then returns a life table using a configurable radix.
"""
