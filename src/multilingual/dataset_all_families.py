"""dataset_all_families.py — Extended multilingual dataset including F1–F5.

This module keeps the original dataset untouched and builds a parallel dataset
for the new multilingual extension where F2–F5 translations are available.

Total rows expected:
  - English base: 288
  - Multilingual F1: 192
  - Multilingual F2–F5: 768
  = 1248 rows
"""

import pandas as pd

import dataset as ds
import ml_sentences as ml_f1
import ml_sentences_f2f5 as ml_f2f5


def _factor_to_domain() -> dict:
    return {f["factor_id"]: f["domain_id"] for f in ds.FACTORS}


def build_ml_f2f5_dataframe() -> pd.DataFrame:
    """Return DataFrame with multilingual F2/F3/F4/F5 sentences.

    The rows are tagged with split='ml' and language in {de, es, ar, zh}.
    """
    factor_domain = _factor_to_domain()
    rows = []

    for factor_id, fam_dict in ml_f2f5.ML_SENTENCES_F2F5.items():
        domain_id = factor_domain[factor_id]
        for family, lang_dict in fam_dict.items():
            for lang, sents in lang_dict.items():
                for idx, text in enumerate(sents):
                    rows.append(
                        {
                            "sentence_id": f"{factor_id}_ml_{family}_{lang}_{idx:02d}",
                            "factor_id": factor_id,
                            "domain_id": domain_id,
                            "surface_family": family,
                            "split": "ml",
                            "language": lang,
                            "sentence_text": text,
                        }
                    )

    return pd.DataFrame(rows)


def build_full_dataframe_all_families() -> pd.DataFrame:
    """Return combined English + multilingual dataframe with F1–F5 coverage."""
    df_en = ds.build_dataframe()
    df_ml_f1 = ml_f1.build_ml_dataframe()
    df_ml_f2f5 = build_ml_f2f5_dataframe()

    df = pd.concat([df_en, df_ml_f1, df_ml_f2f5], ignore_index=True, sort=False)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    df = build_full_dataframe_all_families()
    print(f"rows={len(df)}")
    print(df.groupby(["language", "surface_family"]).size().unstack(fill_value=0).to_string())
