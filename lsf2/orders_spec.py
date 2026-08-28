"""
Parses the --orders CLI argument into a sorted list of unique ints.
Accepts, and any comma-separated mixture of:
  - a single order:        "85"
  - a range (inclusive):   "80-95"
  - a list:                "80,85,90"
  - combinations:          "80-85,90,95-97"
"""
from __future__ import annotations


def parse_orders(spec: str) -> list[int]:
    orders: set[int] = set()
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token[1:]:   # avoid mis-parsing a leading negative sign
            lo_str, hi_str = token.split('-', 1)
            lo, hi = int(lo_str), int(hi_str)
            if lo > hi:
                lo, hi = hi, lo
            orders.update(range(lo, hi + 1))
        else:
            orders.add(int(token))
    if not orders:
        raise ValueError(f"Could not parse any orders from '{spec}'")
    return sorted(orders)
