import re
from typing import Dict, Optional, Tuple, Mapping

import torch

# -----------------------------
# Objectives (single metrics)
# -----------------------------


def stiffness_objective(CH: torch.Tensor) -> torch.Tensor:
    """Mechanical objective (maximize): mean(Ex, Ey, Ez), where E = 1 / S_ii and S = inv(C)."""
    C = 0.5 * (CH + CH.T)
    I6 = torch.eye(6, device=C.device, dtype=C.dtype)
    S = torch.linalg.inv(C + 1e-9 * I6)
    Ex = 1.0 / S[0, 0]
    Ey = 1.0 / S[1, 1]
    Ez = 1.0 / S[2, 2]
    return (Ex + Ey + Ez) / 3.0


def shear_objective(CH: torch.Tensor) -> torch.Tensor:
    """Mechanical objective (maximize): mean(Gyz, Gxz, Gxy), where G = 1 / S_shear."""
    C = 0.5 * (CH + CH.T)
    I6 = torch.eye(6, device=C.device, dtype=C.dtype)
    S = torch.linalg.inv(C + 1e-9 * I6)
    Gyz = 1.0 / S[3, 3]
    Gxz = 1.0 / S[4, 4]
    Gxy = 1.0 / S[5, 5]
    return (Gyz + Gxz + Gxy) / 3.0


def thermal_objective(kappa: torch.Tensor) -> torch.Tensor:
    """Thermal objective (maximize): trace(kappa) = kxx + kyy + kzz.

    kappa: (3,3) or (...,3,3)
    """
    if kappa.dim() < 2 or kappa.shape[-1] != kappa.shape[-2]:
        raise ValueError(f"kappa must be square, got {tuple(kappa.shape)}")
    diag = torch.diagonal(kappa, dim1=-2, dim2=-1)
    return diag.sum()


# -----------------------------
# Objective spec parsing (multi-objective weighted sum)
# -----------------------------
# Supported canonical objectives:
#   - stiffness : use CH (6x6)
#   - shear     : use CH (6x6)
#   - thermal   : use kappa (3x3)
#
# Accepted spec formats (case-insensitive, spaces ignored):
#   - "stiffness"                          -> {stiffness: 1}
#   - "stiffness,thermal"                  -> {stiffness: 1, thermal: 1}
#   - "stiffness:0.4,thermal:0.6"          -> {stiffness: 0.4, thermal: 0.6}
#   - "0.4*stiffness+0.6*thermal"          -> {stiffness: 0.4, thermal: 0.6}
#   - "shear=0.3+thermal=0.7"              -> {shear: 0.3, thermal: 0.7}
#
# Notes:
# - Weights do NOT have to sum to 1 (we use exactly what you give).
# - If you enable normalization (see objective_from_spec), the optimization uses
#   scaled terms but you can still log the raw physical terms.


_ALIASES: Dict[str, str] = {
    # stiffness
    "stiff": "stiffness",
    "stiffness": "stiffness",
    "e": "stiffness",
    "young": "stiffness",
    "youngs": "stiffness",
    # shear
    "shear": "shear",
    "g": "shear",
    # thermal
    "thermal": "thermal",
    "kappa": "thermal",
    "conductivity": "thermal",
    "heat": "thermal",
}

_CANONICAL = ("stiffness", "shear", "thermal")


def _canonicalize_obj_name(name: str) -> str:
    n = str(name).strip().lower()
    n = _ALIASES.get(n, n)
    if n not in _CANONICAL:
        raise ValueError(f"Unknown objective term: {name!r}. Supported: {list(_CANONICAL)}")
    return n


def parse_objective_spec(spec: str) -> Dict[str, float]:
    """Parse objective spec string into {term: weight}.

    See the header comment for accepted formats.
    """
    if spec is None:
        return {"shear": 1.0}
    s = str(spec).strip()
    if not s:
        return {"shear": 1.0}

    # Try single-objective first (for backward compatibility).
    try:
        single = _canonicalize_obj_name(s)
        return {single: 1.0}
    except Exception:
        pass

    # Multi-objective: split by ',' or '+' (spaces ignored).
    s2 = s.replace(" ", "")
    parts = [p for p in re.split(r"[,+]", s2) if p]
    if not parts:
        raise ValueError(f"Invalid objective spec: {spec!r}")

    def _is_num(x: str) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    out: Dict[str, float] = {}
    for part in parts:
        name: str
        w: float

        if ":" in part:
            name, w_str = part.split(":", 1)
            w = float(w_str)
        elif "=" in part:
            name, w_str = part.split("=", 1)
            w = float(w_str)
        elif "*" in part:
            a, b = part.split("*", 1)
            if _is_num(a) and not _is_num(b):
                w = float(a)
                name = b
            elif _is_num(b) and not _is_num(a):
                w = float(b)
                name = a
            else:
                raise ValueError(
                    f"Invalid term {part!r} in objective spec {spec!r}. "
                    f"Expected something like '0.4*stiffness' or 'stiffness:0.4'."
                )
        else:
            name = part
            w = 1.0

        term = _canonicalize_obj_name(name)
        out[term] = out.get(term, 0.0) + float(w)

    if not out:
        raise ValueError(f"Invalid objective spec: {spec!r}")
    return out


def objective_requires(obj_spec: str) -> Tuple[bool, bool]:
    """Return (need_CH, need_kappa)."""
    terms = parse_objective_spec(obj_spec)
    need_ch = any(k in ("stiffness", "shear") for k in terms)
    need_kappa = any(k == "thermal" for k in terms)
    return need_ch, need_kappa


def objective_from_tensor(T: torch.Tensor, obj: str) -> torch.Tensor:
    """Single-objective helper (backward compatible)."""
    o = _canonicalize_obj_name(obj)
    if o == "stiffness":
        return stiffness_objective(T)
    if o == "shear":
        return shear_objective(T)
    if o == "thermal":
        return thermal_objective(T)
    raise ValueError(f"Unknown objective: {obj}")


def objective_from_spec(
    *,
    obj: str,
    CH: Optional[torch.Tensor] = None,
    kappa: Optional[torch.Tensor] = None,
    normalize_by: Optional[Mapping[str, float]] = None,
    return_parts: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]] | torch.Tensor:
    """Compute objective for a spec. Supports weighted multi-objective.

    Parameters
    ----------
    obj:
        Objective spec string (single or multi). Examples:
          - 'shear'
          - 'stiffness:0.4,thermal:0.6'
          - '0.4*stiffness+0.6*thermal'
    CH:
        Homogenized elasticity tensor (6x6), required if spec includes stiffness/shear.
    kappa:
        Homogenized conductivity tensor (3x3), required if spec includes thermal.
    normalize_by:
        Optional scaling dict {term: scale}. If provided, optimization uses (term_value / scale)
        for that term. (Raw term values are still returned in parts_raw.)
    return_parts:
        If True, returns (J, parts_raw, parts_used). Otherwise returns J only.

    Returns
    -------
    J:
        Weighted sum objective (scalar torch.Tensor).
    parts_raw:
        Raw physical metric values for each used term.
    parts_used:
        Values actually used in weighted sum (after optional normalization).
    """
    terms = parse_objective_spec(obj)

    parts_raw: Dict[str, torch.Tensor] = {}
    parts_used: Dict[str, torch.Tensor] = {}
    J = None

    for term, w in terms.items():
        if term in ("stiffness", "shear"):
            if CH is None:
                raise ValueError(f"Objective '{obj}' requires CH (mechanical), but CH=None")
            raw = stiffness_objective(CH) if term == "stiffness" else shear_objective(CH)
        elif term == "thermal":
            if kappa is None:
                raise ValueError(f"Objective '{obj}' requires kappa (thermal), but kappa=None")
            raw = thermal_objective(kappa)
        else:
            raise ValueError(f"Unknown objective term: {term}")

        parts_raw[term] = raw
        if normalize_by is not None and term in normalize_by:
            scale = float(normalize_by[term])
            scale = max(abs(scale), 1e-12)
            used = raw / scale
        else:
            used = raw
        parts_used[term] = used

        contrib = float(w) * used
        J = contrib if J is None else (J + contrib)

    if J is None:
        raise ValueError(f"Empty objective spec: {obj!r}")

    if return_parts:
        return J, parts_raw, parts_used
    return J


# Backward-compatible alias (existing code imports objective_from_CH)
objective_from_CH = objective_from_tensor
