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


# ======================================================================
# SESSION PROJECT EXPORT / IMPORT
# ======================================================================
# The fluid registry and tuning results live only in st.session_state and
# are lost when the browser tab closes. A real PVT study spans days, so
# these helpers serialize the whole working session to a single JSON
# "project" file the user can download and reload later.

PROJECT_FORMAT_VERSION = 1


def export_session(fluid_registry, tuning_results=None, settings=None,
                    indent=2):
    """Serialize a whole working session to a JSON project string.

    fluid_registry  : the dict of saved fluid records
                      (st.session_state['fluid_registry']).
    tuning_results  : optional dict of stored tuning results, keyed by
                      branch (e.g. oil_tune_result, comp_tune_result).
    settings        : optional dict of session settings (unit system,
                      reservoir P/T, etc.).

    Returns a JSON string with a small header so the format can be
    recognised and version-checked on import.
    """
    payload = {
        "_format": "pvt_studio_project",
        "_version": PROJECT_FORMAT_VERSION,
        "fluid_registry": fluid_registry or {},
        "tuning_results": tuning_results or {},
        "settings": settings or {},
    }
    return json.dumps(payload, indent=indent, default=_json_default)


def import_session(text):
    """Parse a JSON project string produced by export_session.

    Returns a dict with keys fluid_registry, tuning_results, settings,
    plus 'ok' (bool) and 'message'. The function is defensive — a file
    that is not a recognised project, or is a bare list/record, is
    reported rather than raising.
    """
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as e:
        return {"ok": False, "message": f"Not valid JSON: {e}",
                "fluid_registry": {}, "tuning_results": {}, "settings": {}}

    # A bare fluid record or list of records — accept as registry-only.
    if isinstance(data, list):
        reg = {}
        for rec in data:
            if isinstance(rec, dict) and "name" in rec:
                reg[rec["name"]] = rec
        return {"ok": True,
                "message": f"Imported {len(reg)} fluid(s) from a fluid "
                           f"list (no full project header).",
                "fluid_registry": reg, "tuning_results": {},
                "settings": {}}
    if isinstance(data, dict) and data.get("_format") != "pvt_studio_project":
        # Maybe a single fluid record.
        if "name" in data and "fluid_type" in data:
            return {"ok": True,
                    "message": "Imported a single fluid record.",
                    "fluid_registry": {data["name"]: data},
                    "tuning_results": {}, "settings": {}}
        return {"ok": False,
                "message": "JSON is not a recognised PVT Studio project "
                           "file.",
                "fluid_registry": {}, "tuning_results": {}, "settings": {}}

    ver = data.get("_version", 0)
    msg = f"Project loaded (format v{ver})."
    if ver > PROJECT_FORMAT_VERSION:
        msg += (" Note: the file was written by a newer version of the "
                "app — some data may not load.")
    return {"ok": True, "message": msg,
            "fluid_registry": data.get("fluid_registry", {}),
            "tuning_results": data.get("tuning_results", {}),
            "settings": data.get("settings", {})}
