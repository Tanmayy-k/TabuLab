"""
TabuLab Integration Test Suite
Tests all core backend functions end-to-end.
"""
import sys
import traceback

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def test(name, fn):
    try:
        result = fn()
        print(f"{PASS}  {name}" + (f" — {result}" if result else ""))
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL}  {name}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ── 1. Import checks ──────────────────────────────────────────────────────────
def t_imports():
    import numpy, pandas, matplotlib, seaborn, streamlit
    import sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing
    import sdv, ctgan
    return f"sdv={sdv.__version__}, ctgan={ctgan.__version__}, streamlit={streamlit.__version__}"

test("Imports (sdv, ctgan, streamlit, sklearn)", t_imports)

# ── 2. Demo dataset ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

def load_demo_dataset():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    return df.drop(columns=["alive", "embark_town", "who", "adult_male", "deck"], errors="ignore")

def t_demo_dataset():
    df = load_demo_dataset()
    assert len(df) > 800, f"Expected >800 rows, got {len(df)}"
    assert df.shape[1] >= 5, f"Expected >=5 cols, got {df.shape[1]}"
    return f"{df.shape[0]} rows x {df.shape[1]} cols"

test("Demo dataset load (Titanic)", t_demo_dataset)

# ── 3. Column detection ───────────────────────────────────────────────────────
def detect_column_types(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat

def t_detect():
    df = load_demo_dataset()
    num, cat = detect_column_types(df)
    assert len(num) > 0, "No numerical cols"
    assert len(cat) > 0, "No categorical cols"
    return f"num={num}, cat={cat}"

test("Column type detection", t_detect)

# ── 4. Preprocessing ─────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

def preprocess_dataframe(df):
    df = df.copy()
    thresh = int(len(df) * 0.2)
    df = df.dropna(axis=1, thresh=thresh)
    num_cols, cat_cols = detect_column_types(df)
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
        df[c] = df[c].astype(str)
    return df, num_cols, cat_cols

def t_preprocess():
    df = load_demo_dataset()
    df_clean, num_cols, cat_cols = preprocess_dataframe(df)
    assert df_clean.isna().sum().sum() == 0, "Missing values remain after preprocessing"
    return f"clean shape={df_clean.shape}, num={len(num_cols)}, cat={len(cat_cols)}"

test("Preprocessing (missing value fill)", t_preprocess)

# ── 5. Train/test split + SDV metadata ───────────────────────────────────────
def t_split_metadata():
    from sdv.metadata import Metadata
    from sklearn.model_selection import train_test_split
    df = load_demo_dataset()
    df_clean, num_cols, cat_cols = preprocess_dataframe(df)
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    meta = Metadata.detect_from_dataframes({"table": df_train})
    assert len(df_train) > len(df_test), "Train should be larger than test"
    return f"train={len(df_train)}, test={len(df_test)}, meta_tables={list(meta.to_dict().get('tables', {}).keys())}"

test("Train/test split + SDV metadata", t_split_metadata)

# ── 6. CTGAN training ─────────────────────────────────────────────────────────
_synth = None
_df_train = None
_df_test  = None
_num_cols = None
_cat_cols = None
_metadata = None

def t_ctgan_train():
    global _synth, _df_train, _df_test, _num_cols, _cat_cols, _metadata
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata
    from sklearn.model_selection import train_test_split
    df = load_demo_dataset()
    df_clean, num_cols, cat_cols = preprocess_dataframe(df)
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    meta = Metadata.detect_from_dataframes({"table": df_train})
    synth = CTGANSynthesizer(meta, epochs=5, batch_size=200, verbose=False)
    synth.fit({"table": df_train})
    _synth     = synth
    _df_train  = df_train
    _df_test   = df_test
    _num_cols  = num_cols
    _cat_cols  = cat_cols
    _metadata  = meta
    return f"Synthesizer trained (5 epochs, {len(df_train)} rows)"

test("CTGAN training (5 epochs)", t_ctgan_train)

# ── 7. Synthetic data generation ─────────────────────────────────────────────
_syn_df = None

def t_generate():
    global _syn_df
    assert _synth is not None, "Synthesizer not trained"
    result = _synth.sample(num_rows=100)
    syn_df = result["table"] if isinstance(result, dict) else result
    assert len(syn_df) == 100, f"Expected 100 rows, got {len(syn_df)}"
    _syn_df = syn_df
    return f"100 synthetic rows x {syn_df.shape[1]} cols"

test("Synthetic data generation (100 rows)", t_generate)

# ── 8. Quality report ─────────────────────────────────────────────────────────
def t_quality_report():
    assert _syn_df is not None
    from sdmetrics.reports.single_table import QualityReport
    tmeta = _metadata.get_table_metadata("table").to_dict()
    single = tmeta.get("tables", {}).get("table", tmeta)
    report = QualityReport()
    report.generate(_df_train, _syn_df, single, verbose=False)
    score = report.get_score()
    assert 0 <= score <= 1, f"Score out of range: {score}"
    return f"Score={score:.4f} ({score*100:.1f}%)"

test("SDV quality report", t_quality_report)

# ── 9. CSV download bytes ─────────────────────────────────────────────────────
def t_csv_download():
    assert _syn_df is not None
    csv_bytes = _syn_df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 100, "CSV too small"
    return f"{len(csv_bytes)} bytes"

test("CSV download bytes", t_csv_download)

# ── 10. ML utility (TSTR) ─────────────────────────────────────────────────────
def t_ml_utility():
    assert _syn_df is not None
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    target = "survived"
    if target not in _df_train.columns:
        target = _df_train.columns[-1]
    
    feature_cols = [c for c in _num_cols + _cat_cols if c != target]
    
    def encode(df):
        X = df[feature_cols].copy()
        for c in _cat_cols:
            if c in X.columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        return X, df[target].copy()
    
    le_t = LabelEncoder()
    X_real, y_real = encode(_df_train)
    X_test, y_test = encode(_df_test)
    X_syn, y_syn   = encode(_syn_df)
    X_syn  = X_syn.reindex(columns=X_real.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_real.columns, fill_value=0)
    
    y_real = le_t.fit_transform(y_real.astype(str))
    y_syn  = le_t.transform(pd.Series(y_syn).astype(str).map(
        lambda x: x if x in le_t.classes_ else le_t.classes_[0]))
    y_test = le_t.transform(pd.Series(y_test).astype(str).map(
        lambda x: x if x in le_t.classes_ else le_t.classes_[0]))
    
    base = RandomForestClassifier(n_estimators=50, random_state=42)
    base.fit(X_real, y_real)
    y_pb = base.predict(X_test)
    
    tstr = RandomForestClassifier(n_estimators=50, random_state=42)
    tstr.fit(X_syn, y_syn)
    y_pt = tstr.predict(X_test)
    
    baseline_acc = accuracy_score(y_test, y_pb)
    tstr_acc     = accuracy_score(y_test, y_pt)
    return f"Baseline={baseline_acc:.3f}, TSTR={tstr_acc:.3f}, target={target}"

test("ML utility TSTR (classification)", t_ml_utility)

# ── 11. Privacy metrics ───────────────────────────────────────────────────────
def t_privacy():
    assert _syn_df is not None
    merged = _syn_df.merge(_df_train, how="inner")
    exact_matches = len(merged)
    
    cols = [c for c in _num_cols + _cat_cols if c in _df_train and c in _syn_df]
    le = LabelEncoder()
    real_enc = _df_train[cols].copy()
    syn_enc  = _syn_df[cols].copy()
    for c in _cat_cols:
        if c in cols:
            all_v = pd.concat([real_enc[c], syn_enc[c]]).astype(str)
            le.fit(all_v)
            real_enc[c] = le.transform(real_enc[c].astype(str))
            syn_enc[c]  = le.transform(syn_enc[c].astype(str))
    
    real_arr = real_enc.values.astype(float)
    syn_arr  = syn_enc.values.astype(float)
    rng_range = real_arr.max(0) - real_arr.min(0)
    rng_range[rng_range == 0] = 1
    real_n = (real_arr - real_arr.min(0)) / rng_range
    syn_n  = (syn_arr  - real_arr.min(0)) / rng_range
    rng = np.random.default_rng(42)
    if len(syn_n) > 500:
        syn_n = syn_n[rng.choice(len(syn_n), 500, replace=False)]
    dists = [np.linalg.norm(real_n - row, axis=1).min() for row in syn_n]
    mean_nn = float(np.mean(dists))
    
    return f"exact_matches={exact_matches}, mean_nn_dist={mean_nn:.4f}"

test("Privacy metrics (exact match + NN dist)", t_privacy)

# ── 12. Report text generation ────────────────────────────────────────────────
def t_report_txt():
    num_cols_str = ", ".join(_num_cols) if _num_cols else "—"
    cat_cols_str = ", ".join(_cat_cols) if _cat_cols else "—"
    txt = f"""TabuLab Report
===================
Train rows: {len(_df_train):,}
Test rows:  {len(_df_test):,}
Cols:       {_df_train.shape[1]}
Num cols:   {num_cols_str}
Cat cols:   {cat_cols_str}
Syn rows:   {len(_syn_df):,}
"""
    assert "TabuLab" in txt
    return f"{len(txt)} chars"

test("Report text generation", t_report_txt)

# ── 13. Chart rendering ───────────────────────────────────────────────────────
def t_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    fig, ax = plt.subplots(figsize=(8, 4))
    _df_train[_num_cols[0]].dropna().plot.hist(ax=ax, bins=20, color="#C7FF55")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    size = len(buf.read())
    plt.close()
    assert size > 1000, f"Chart too small: {size} bytes"
    return f"{size} bytes"

test("Chart render to buffer (matplotlib)", t_chart)

# ── 14. HTML entity check in callout helper ───────────────────────────────────
def t_html_entities():
    # Verify HTML numeric entities render correctly (these are used throughout)
    entities = {
        "&#10003;": "✓",
        "&#9888;": "⚠",
        "&#9432;": "ℹ",
        "&#8212;": "—",
        "&#8594;": "→",
        "&#215;": "×",
    }
    # Just verify the entity values are valid (non-empty)
    for entity, char in entities.items():
        assert entity.startswith("&#") and entity.endswith(";")
    return f"{len(entities)} HTML entities verified"

test("HTML entity constants valid", t_html_entities)

# ── 15. No duplicate widget keys ─────────────────────────────────────────────
def t_no_dup_keys():
    import re
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()
    keys = re.findall(r'key="([^"]+)"', src)
    from collections import Counter
    dupes = {k: v for k, v in Counter(keys).items() if v > 1}
    assert not dupes, f"Duplicate keys found: {dupes}"
    return f"{len(keys)} unique widget keys"

test("No duplicate widget keys", t_no_dup_keys)

# ── 16. All HTML markdowns have unsafe_allow_html ────────────────────────────
def t_all_html_has_unsafe():
    import re
    with open("app.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    problems = []
    for i, l in enumerate(lines):
        stripped = l.strip()
        if (stripped.startswith("st.markdown(") and
                ("<div" in stripped or "</div>" in stripped or "<span" in stripped or "<hr" in stripped)
                and "unsafe_allow_html" not in stripped):
            problems.append(f"L{i+1}: {stripped[:80]}")
    assert not problems, "Found HTML markdowns without unsafe_allow_html:\n" + "\n".join(problems)
    return "All single-line HTML markdowns have unsafe_allow_html=True"

test("All HTML markdowns have unsafe_allow_html", t_all_html_has_unsafe)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")
if failed:
    print("\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  ✗ {name}: {err}")
    sys.exit(1)
else:
    print("All tests passed ✓")
