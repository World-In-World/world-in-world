import argparse


def _coerce(v: str):
    """Best-effort lightweight type coercion; falls back to str."""
    vl = v.lower()
    if vl in ("true", "false"):        # booleans
        return vl == "true"
    try:                               # int
        return int(v)
    except ValueError:
        pass
    try:                               # float
        return float(v)
    except ValueError:
        pass
    return v                           # leave as str


def _push(d, k, v):
    """Accumulate possibly repeated keys into list."""
    if k in d:
        if not isinstance(d[k], list):
            d[k] = [d[k]]
        d[k].append(v)
    else:
        d[k] = v


def parse_extra_cli(unknown):
    """
    Convert unknown argv tokens (list[str]) into {k: v}.
    Accepts --k=v, --k v, and bare --flag.
    """
    extra = {}
    pending_key = None
    for tok in unknown:
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:                 # --k=v
                k, v = body.split("=", 1)
                _push(extra, k, _coerce(v))
                pending_key = None
            else:                           # --k  (value may come next)
                # if a previous --k had no value, treat it as flag True
                if pending_key is not None:
                    _push(extra, pending_key, True)
                pending_key = body
        else:
            # value for the most recent --k
            if pending_key is None:
                raise ValueError(f"Unexpected token '{tok}' without a preceding --key")
            _push(extra, pending_key, _coerce(tok))
            pending_key = None
    # trailing bare --k
    if pending_key is not None:
        _push(extra, pending_key, True)
    return extra
