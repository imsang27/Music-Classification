"""Deterministic artist assignment and disjoint audio-manifest checks."""

import hashlib


def artist_partition(artist_id, seed):
    if not isinstance(artist_id, str) or not artist_id or not isinstance(seed, str) or not seed:
        raise ValueError("Artist identifier and split seed required")
    digest = hashlib.sha256((seed + ":artist:" + artist_id).encode("utf-8")).digest()
    return "calibration" if digest[0] % 2 == 0 else "holdout"


def audit_partitions(calibration, holdout, excluded):
    """Fail on track, artist, album or exact-byte audio overlap.

    This does not detect covers, re-encodings or prior model training overlap.
    """
    fields = ("track_id", "artist_id", "album_id", "audio_sha256")
    groups = {"calibration": list(calibration), "holdout": list(holdout), "excluded": list(excluded)}
    for name in ("calibration", "holdout"):
        rows = groups[name]
        for row in rows:
            if any(not isinstance(row.get(field), str) or not row[field] for field in fields):
                raise ValueError(f"Missing identity fields in {name}")
        for field in fields:
            values = [r[field] for r in rows]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {field} within {name}")
    report = {}
    for first, second in [("calibration", "holdout"), ("calibration", "excluded"), ("holdout", "excluded")]:
        counts = {}
        for field in fields:
            a = {r[field] for r in groups[first] if r.get(field)}
            b = {r[field] for r in groups[second] if r.get(field)}
            overlap = a & b
            if overlap:
                raise ValueError(f"{field} overlap: {first}/{second}")
            counts[field] = len(overlap)
        report[first + "_vs_" + second] = counts
    return report
