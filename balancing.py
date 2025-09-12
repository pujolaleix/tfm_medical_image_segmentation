import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import random
import pandas as pd
import numpy as np


def injury_label_remap(df: pd.DataFrame, mapping: pd.DataFrame):

    mapping_df =  mapping[['original_label', 'target_superclass']]
    out_df = pd.merge(df, mapping_df, left_on="injury_label", right_on="original_label", how="left")
    out_df['injury_label'] = out_df.pop('target_superclass')

    return out_df




def _resolve_label(sample: Dict[str, Any], label_key: str = "case_label") -> Optional[str]:
    """
    Get a label for balancing. Prefer top-level `case_label`.
    If missing/empty, fallback to majority label among regions[].label.
    """
    val = sample.get(label_key, None)
    if val not in (None, "", [], {}):
        return str(val)
    regs = sample.get("regions", [])
    if isinstance(regs, list) and regs:
        labs = [r.get("label") for r in regs if r.get("label") not in (None, "", [], {})]
        if labs:
            return Counter(map(str, labs)).most_common(1)[0][0]
    return None




def downsample_for_balancing(
    train_s: List[Dict[str, Any]],
    label_key: str = "case_label",
    skew: float = 0.0,                 # 0.0 = full balance, 1.0 = no balance
    max_ratio: Optional[float] = None, # e.g., 2.0 => majority ≤ 2× minority after downsampling
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Downsample a structured training set (list of dicts) by class using `skew` and optional `max_ratio`.
    Returns a new list in the same structured format (each element is the original dict).
    This is intended for the **training split only**; keep val/test untouched.

    Each item in `train_s` is expected to have:
      - image_id, image_path, width, height, patient_id, case_label, regions[...]
    """
    assert 0.0 <= skew <= 1.0, f"`skew` must be in [0,1], got {skew}"
    if max_ratio is not None:
        assert max_ratio >= 1.0, f"`max_ratio` must be ≥ 1, got {max_ratio}"

    # Build a minimal frame for counting/sampling
    labels = []
    idxs = []
    for i, s in enumerate(train_s):
        lab = _resolve_label(s, label_key=label_key)
        if lab is None:
            raise ValueError(f"Sample idx={i} has no resolvable label "
                             f"(missing `{label_key}` and empty regions[].label).")
        labels.append(lab)
        idxs.append(i)

    df = pd.DataFrame({"idx": idxs, "label": labels})

    counts_before = df["label"].value_counts().sort_index()
    if verbose:
        print("[Balancing] Class counts BEFORE:\n", counts_before.to_string())

    # Compute target counts per class:
    # (1 - skew) * n_min + skew * n_orig
    n_orig = counts_before
    n_min = int(n_orig.min())
    targets = ((1.0 - skew) * n_min + skew * n_orig.values).round().astype(int)
    targets = np.maximum(targets, 1)
    target_map = dict(zip(n_orig.index.tolist(), targets.tolist()))

    # Optional majority/minority cap
    if max_ratio is not None:
        current_min_target = max(1, int(min(target_map.values())))
        cap = int(np.floor(max_ratio * current_min_target))
        for k in target_map:
            target_map[k] = min(target_map[k], cap)

    # Sample indices per class
    rng = random.Random(seed)
    keep_idx: List[int] = []
    for lbl, grp in df.groupby("label"):
        want = int(target_map[lbl])
        pool = grp["idx"].tolist()
        if len(pool) <= want:
            keep_idx.extend(pool)
        else:
            keep_idx.extend(rng.sample(pool, want))

    rng.shuffle(keep_idx)
    df_after = df[df["idx"].isin(keep_idx)]
    counts_after = df_after["label"].value_counts().sort_index()
    if verbose:
        print("[Balancing] Class counts AFTER:\n", counts_after.to_string())

    # Return same structured format
    return [train_s[i] for i in keep_idx]




