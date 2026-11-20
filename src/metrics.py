import numpy as np


def _check_signal(x, name):
    x = np.asarray(x)
    if np.iscomplexobj(x) or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"{name} must contain real numbers, got dtype {x.dtype}")
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D signal, got shape {x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must not be empty")
    x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return x


def snr_db(clean, test):
    """SNR of test relative to clean in dB. Returns +inf when test equals clean."""
    clean = _check_signal(clean, "clean")
    test = _check_signal(test, "test")
    if len(clean) != len(test):
        raise ValueError(
            f"clean and test must have the same length, got {len(clean)} and {len(test)}"
        )

    signal_energy = np.sum(clean**2)
    if signal_energy == 0:
        raise ValueError("clean has zero energy; SNR is undefined")

    error_energy = np.sum((clean - test) ** 2)
    if error_energy == 0:
        return float("inf")
    return float(10 * np.log10(signal_energy / error_energy))
