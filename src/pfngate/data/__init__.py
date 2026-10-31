"""Loaders for real data. Real data is test only: see docs/DAPHNIA.md and docs/GRID.md."""

REAL_SPLIT = "real_tipping"  # Daphnia populations: the real tipping data
REAL_CONTROL_SPLIT = "real_control"  # grid frequency in normal operation: false alarms only
REAL_SPLITS = (REAL_SPLIT, REAL_CONTROL_SPLIT)  # every real split, all of them test only
