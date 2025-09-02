import pandas as pd


def injury_label_remap(df: pd.DataFrame, mapping: pd.DataFrame):

    mapping_df =  mapping[['original_label', 'target_superclass']]
    out_df = pd.merge(df, mapping_df, left_on="injury_label", right_on="original_label", how="left")
    out_df['injury_label'] = out_df.pop('target_superclass')

    return out_df


def downsample_to_min_per_class(df: pd.DataFrame, label_col: str = "injury_label", random_state: int = 42) -> pd.DataFrame:
    # Compute N as the size of the smallest class
    counts = df[label_col].value_counts(dropna=False)
    N = int(counts.min())

    # Sample N per class (no replacement), then shuffle
    balanced = (
        df.groupby(label_col, group_keys=False)
          .apply(lambda g: g.sample(n=N, replace=False, random_state=random_state))
          .sample(frac=1.0, random_state=random_state)  # shuffle rows
          .reset_index(drop=True)
    )

    # Optional: quick sanity check
    # print(balanced[label_col].value_counts())

    return balanced




