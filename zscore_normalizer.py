import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List

from signals import Discharge, SignalType

# -----------------------------------------------------------------------------
# Z-SCORE NORMALISATION UTILITIES (per Sensor / SignalType)
# -----------------------------------------------------------------------------
# We keep all functions stateless; the stats (mean, std) are passed around as
# plain python dicts so that they can be serialised along with the model and
# reused at inference time without touching global state.
# -----------------------------------------------------------------------------

StatsDict = Dict[SignalType, Tuple[float, float]]  #  {SignalType: (mean, std)}


def compute_zscore_stats(discharges: List[Discharge]) -> StatsDict:
    """Compute global mean and std **per SignalType** across a list of discharges.

    Parameters
    ----------
    discharges : list[Discharge]
        All discharges in the *training* split.

    Returns
    -------
    StatsDict
        Mapping from SignalType to (μ, σ).
    """
    # Accumulate sum and sum-of-squares per type
    accum = defaultdict(lambda: {
        "count": 0,
        "sum": 0.0,
        "sum2": 0.0,
    })

    for disc in discharges:
        for sig in disc.signals:
            stype = sig.signal_type
            vals = np.asarray(sig.values, dtype=np.float64)
            a = accum[stype]
            a["count"] += vals.size
            a["sum"] += vals.sum()
            a["sum2"] += np.square(vals).sum()

    stats: StatsDict = {}
    for stype, a in accum.items():
        n = a["count"] if a["count"] > 0 else 1
        mean = a["sum"] / n
        # unbiased std (ddof=0)  – small diff at this scale
        var = a["sum2"] / n - mean ** 2
        std = float(np.sqrt(max(var, 1e-12)))  # avoid div/0 later
        stats[stype] = (float(mean), std)
    return stats


def apply_zscore(discharges: List[Discharge], stats: StatsDict) -> List[Discharge]:
    """Apply in-place z-score normalisation to each signal.

    Any SignalType missing in *stats* is left unchanged (rare, but keeps the
    function safe if new sensors appear at test time).
    """
    for disc in discharges:
        for sig in disc.signals:
            if sig.signal_type not in stats:
                continue  # unseen sensor type – skip normalisation
            mean, std = stats[sig.signal_type]
            sig.values = ((np.asarray(sig.values, dtype=np.float32) - mean) / std).tolist()
        disc.is_normalized = True
    return discharges


def are_zscored(discharges: List[Discharge], stats: StatsDict, atol: float = 1e-2) -> bool:
    """Heuristic check: sample a few signals and see if their mean≈0, std≈1.

    *atol* is the absolute tolerance for the mean.
    """
    sample = 0
    for disc in discharges:
        for sig in disc.signals:
            if sig.signal_type in stats:
                vals = np.asarray(sig.values)
                if abs(vals.mean()) > atol or abs(vals.std() - 1) > 0.2:  # loose check
                    return False
                sample += 1
                if sample >= 5:  # we don't need to check everything
                    return True
    return sample > 0

# -----------------------------------------------------------------------------
# Convenience wrapper combining both steps (for small datasets)
# -----------------------------------------------------------------------------

def zscore_fit_transform(discharges: List[Discharge]) -> Tuple[List[Discharge], StatsDict]:
    """Compute stats on *discharges* and return (normalised copy, stats)."""
    stats = compute_zscore_stats(discharges)
    return apply_zscore(discharges, stats), stats
