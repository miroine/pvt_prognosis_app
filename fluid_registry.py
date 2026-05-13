"""
Fluid save/load registry.

Each saved fluid is a dict with:
    name        : user-given label
    timestamp   : ISO 8601 creation time
    fluid_type  : 'oil' / 'dry_gas' / 'wet_gas' / 'compositional'
    units       : 'Field' / 'SI'
    parameters  : dict of fluid-type-specific inputs (API, SG, composition, etc.)
    tuning      : optional tuning state (correlation Pb shift, EOS multipliers, etc.)

Fluids are serialized as JSON and can be downloaded / uploaded as files.
"""

import json
from datetime import datetime, timezone


def make_fluid_record(name, fluid_type, units, parameters, tuning=None, notes=""):
    return {
        "name":         name,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "fluid_type":   fluid_type,
        "units":        units,
        "parameters":   parameters,
        "tuning":       tuning or {},
        "notes":        notes,
        "app_version":  "PVT Studio v1.0",
    }


def to_json(record_or_list, indent=2):
    """Serialize a single fluid record OR a list of records to JSON text."""
    return json.dumps(record_or_list, indent=indent, default=_json_default)


def from_json(text):
    """Parse JSON text back to a fluid record (or list of records)."""
    return json.loads(text)


def _json_default(o):
    """Fallback for numpy scalars / arrays."""
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except ImportError:
        pass
    if hasattr(o, '__dict__'):
        return o.__dict__
    return str(o)


def summarize(record):
    """Human-readable one-line summary for a saved fluid."""
    p = record.get("parameters", {})
    ft = record.get("fluid_type", "?")
    if ft == "oil":
        return (f"Oil — API={p.get('api', '?')}, "
                f"gas SG={p.get('gas_sg', '?')}, "
                f"Rsi={p.get('Rsi', '?')}")
    if ft == "dry_gas":
        return f"Dry gas — SG={p.get('gas_sg', '?')}"
    if ft == "wet_gas":
        return f"Wet gas — SG={p.get('gas_sg', '?')}, CGR={p.get('cgr', '?')}"
    if ft == "compositional":
        n = len(p.get("composition", {}))
        return f"Compositional — {n} components"
    return ft
