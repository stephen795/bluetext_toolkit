from typing import Iterable, Optional, Dict
from rapidfuzz import fuzz


def normalize_name(name: str) -> str:
    if not name:
        return ''
    return ''.join(ch for ch in name.lower() if ch.isalnum() or ch.isspace()).strip()


def find_best_match(name: str, candidates: Iterable[str], overrides: Optional[Dict[str, str]] = None, threshold: float = 0.8) -> Optional[str]:
    """Find best matching candidate for name.

    - First uses overrides (mapping normalized name -> candidate)
    - Then exact normalized match
    - Then difflib.get_close_matches with cutoff=threshold
    Returns the matched candidate string or None.
    """
    if not name:
        return None
    norm = normalize_name(name)
    # overrides: keys are normalized pbp names or sleeper names mapping to candidate
    if overrides:
        # check direct normalized key
        if norm in overrides:
            return overrides[norm]

    # prepare normalized candidate map
    cand_map = {normalize_name(c): c for c in candidates}
    # exact match
    if norm in cand_map:
        return cand_map[norm]

    # fuzzy match using rapidfuzz token_set_ratio
    best = None
    best_score = 0.0
    for nc, orig in cand_map.items():
        score = fuzz.token_set_ratio(norm, nc) / 100.0
        if score > best_score:
            best_score = score
            best = orig
    if best_score >= threshold:
        return best

    # fallback: try matching on last name token only if threshold not extremely strict
    if threshold < 0.95:
        tokens = norm.split()
        if tokens:
            last = tokens[-1]
            # find candidates with same last token
            for nc, orig in cand_map.items():
                if nc.split() and nc.split()[-1] == last:
                    return orig

    return None
