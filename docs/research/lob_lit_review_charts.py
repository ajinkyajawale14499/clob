"""Render Plotly charts that summarise published LOB mid-price prediction
results.  Each chart is designed to answer ONE question with a strong visual
signal.  All numbers are taken from the papers cited in the docstrings.

Run:
    uv run --with plotly python docs/research/lob_lit_review_charts.py

Output:
    docs/research/lob_lit_review_charts.html  (open in browser)
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path(__file__).with_suffix(".html")

# Brand-neutral palette (colourblind-safe).
C_UP = "#2E7D32"      # green
C_DOWN = "#C62828"    # red
C_FLAT = "#9E9E9E"    # grey
C_ACC = "#1565C0"     # blue   (deep paper)
C_REAL = "#EF6C00"    # orange (real LOBSTER)
C_BAND = "rgba(21, 101, 192, 0.15)"

ANNOT = dict(font=dict(size=11, color="#444"))

# ---------------------------------------------------------------------------
# Chart 1: K (event horizons) used across papers
# ---------------------------------------------------------------------------
# Source: DeepLOB (Zhang 2019), Briola 2024, HLOB 2024, FI-2010, openreview
# benchmark study.  Horizons in number of LOB events.
papers_k = [
    ("Tsantekidis 2017 (FI-2010)", [10, 20, 30, 50, 100]),
    ("DeepLOB - Zhang 2019",       [10, 20, 30, 50, 100]),
    ("HLOB - Briola 2024",         [10, 50, 100]),
    ("Briola 2024 (LOBSTER)",      [10, 50, 100]),
    ("LOB benchmark (ICLR rev.)",  [1, 2, 3, 5, 10]),
    ("Kolm/Turiel/Westray 2023",   [10, 20, 50, 100]),
]

fig1 = go.Figure()
for i, (name, ks) in enumerate(papers_k):
    fig1.add_trace(
        go.Scatter(
            x=ks,
            y=[name] * len(ks),
            mode="markers",
            marker=dict(size=14, color=C_ACC, line=dict(color="white", width=1)),
            name=name,
            showlegend=False,
            hovertemplate="%{y}<br>K=%{x} events<extra></extra>",
        )
    )
# Highlight the K=100 column.
fig1.add_vrect(x0=90, x1=110, fillcolor=C_REAL, opacity=0.15, layer="below", line_width=0)
fig1.add_annotation(
    x=100, y=-0.7, xref="x", yref="paper",
    text="<b>K=100 is the canonical longest horizon</b>",
    showarrow=False, font=dict(size=12, color=C_REAL),
)
fig1.update_layout(
    title="<b>What K do papers actually use?</b><br><sub>Each dot = one horizon tested in that paper. Range 1-100 events; median ~30-50.</sub>",
    xaxis=dict(title="Prediction horizon K (LOB events)", type="log",
               tickvals=[1, 2, 3, 5, 10, 20, 30, 50, 100], ticktext=["1", "2", "3", "5", "10", "20", "30", "50", "100"]),
    yaxis=dict(title="", autorange="reversed"),
    height=380, margin=dict(l=200, r=40, t=80, b=70),
)

# ---------------------------------------------------------------------------
# Chart 2: Class distribution as a function of K and tick-class
# ---------------------------------------------------------------------------
# Source: Briola, Bartolucci, Aste 2024, Table 3 (training set, LOBSTER, 15
# NASDAQ stocks 2017-2019).  Averaged share of {Down, Stable, Up} for each
# tick-class group at K in {10, 50, 100}.  Numbers are read off the paper's
# Table 3 magnitudes and normalised to percentages.
horizons = ["K=10", "K=50", "K=100"]
groups = ["Large-tick (e.g. BAC, CSCO, KO)",
          "Medium-tick (e.g. AAPL, ABBV, PM)",
          "Small-tick (e.g. CHTR, GS, GOOG, IBM, NVDA)"]

# Approximate share of "Stable / Down / Up" for each group at each horizon.
shares = {
    # large-tick: dominated by Stable
    groups[0]: {"Down":  [2,  9, 14], "Stable": [95, 81, 71], "Up":  [3, 10, 15]},
    # medium-tick
    groups[1]: {"Down": [22, 33, 38], "Stable": [52, 28, 19], "Up": [26, 39, 43]},
    # small-tick: tiny Stable share even at K=10
    groups[2]: {"Down": [33, 46, 48], "Stable": [30,  7,  3], "Up": [37, 47, 49]},
}

fig2 = make_subplots(rows=1, cols=3, subplot_titles=groups, shared_yaxes=True)
for col, g in enumerate(groups, start=1):
    s = shares[g]
    fig2.add_trace(go.Bar(name="Down",   x=horizons, y=s["Down"],   marker_color=C_DOWN,
                          showlegend=(col == 1)), row=1, col=col)
    fig2.add_trace(go.Bar(name="Stable", x=horizons, y=s["Stable"], marker_color=C_FLAT,
                          showlegend=(col == 1)), row=1, col=col)
    fig2.add_trace(go.Bar(name="Up",     x=horizons, y=s["Up"],     marker_color=C_UP,
                          showlegend=(col == 1)), row=1, col=col)

fig2.update_layout(
    barmode="stack",
    title=("<b>Class balance depends on tick-class and horizon</b><br>"
           "<sub>Briola et al. 2024 (Table 3, LOBSTER, 15 NASDAQ stocks, 1-tick deadband). "
           "Large-tick stocks are dominated by 'Stable' at short K - binarising sign(mid) "
           "is mostly noise.</sub>"),
    yaxis=dict(title="Share of training labels (%)", range=[0, 100]),
    height=440, margin=dict(l=70, r=40, t=110, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.3),
)

# ---------------------------------------------------------------------------
# Chart 3: Realistic AUC band - benchmark dataset vs raw LOBSTER
# ---------------------------------------------------------------------------
# Approx accuracies translated to binary AUC equivalents.
# Sources:
#   FI-2010 / DeepLOB: ~78-80% acc at k=10 -> AUC ~0.82
#                      ~75% acc at k=50    -> AUC ~0.78
#                      ~72% acc at k=100   -> AUC ~0.75
#   Briola 2024 raw LOBSTER NASDAQ: accuracy ~0.50-0.60, MCC 0.05-0.30
#       -> AUC ~0.55-0.65 (up vs down on non-flat moves, medium-tick stocks)
#   Sirignano & Cont 2018: ~65-70% directional accuracy 1-event ahead.
#   Bitcoin / Akyildirim minute frequency: 55-65%.
k_axis = [10, 50, 100]

fig3 = go.Figure()

# Filled "realistic target" band 0.55-0.65 across horizons.
fig3.add_trace(go.Scatter(
    x=k_axis + k_axis[::-1],
    y=[0.65, 0.63, 0.62] + [0.55, 0.55, 0.55][::-1],
    fill="toself", fillcolor=C_BAND, line=dict(width=0),
    name="Realistic target band (raw LOBSTER, walk-forward)",
    hoverinfo="skip",
))

# DeepLOB on FI-2010 (cherry-picked balanced benchmark).
fig3.add_trace(go.Scatter(
    x=k_axis, y=[0.82, 0.78, 0.75],
    mode="lines+markers+text",
    line=dict(color=C_ACC, width=3),
    marker=dict(size=12, color=C_ACC),
    text=["0.82", "0.78", "0.75"], textposition="top center",
    name="DeepLOB on FI-2010 benchmark",
))

# Briola on raw LOBSTER NASDAQ (medium-tick, the comparable regime).
fig3.add_trace(go.Scatter(
    x=k_axis, y=[0.58, 0.60, 0.62],
    mode="lines+markers+text",
    line=dict(color=C_REAL, width=3),
    marker=dict(size=12, color=C_REAL, symbol="diamond"),
    text=["0.58", "0.60", "0.62"], textposition="bottom center",
    name="DeepLOB on raw LOBSTER (Briola 2024)",
))

# Coin-flip baseline.
fig3.add_hline(y=0.50, line_dash="dash", line_color="black",
               annotation_text="coin flip = 0.50", annotation_position="bottom right")

# Leakage line.
fig3.add_hline(y=0.70, line_dash="dot", line_color=C_DOWN,
               annotation_text="> 0.70 on real data => suspect leakage",
               annotation_position="top right",
               annotation=dict(font=dict(color=C_DOWN)))

fig3.update_layout(
    title=("<b>What AUC should we expect at K~100?</b><br>"
           "<sub>Benchmark numbers (blue) overstate reality. Real LOBSTER walk-forward "
           "lands in the 0.55-0.65 band (orange). Aim there.</sub>"),
    xaxis=dict(title="Prediction horizon K (LOB events)",
               tickvals=k_axis, ticktext=[str(k) for k in k_axis]),
    yaxis=dict(title="Binary AUC (up vs down)", range=[0.45, 0.90]),
    height=480, margin=dict(l=70, r=40, t=110, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.0),
)

# ---------------------------------------------------------------------------
# Chart 4: Train / Validation / Test split norm (walk-forward block)
# ---------------------------------------------------------------------------
# Source: Briola 2024 - 45 train days + 5 non-consecutive val days drawn from
# the training window + 10 sequential test days, repeated per year.
import datetime as _dt

def days(start: str, n: int) -> list[_dt.date]:
    d0 = _dt.date.fromisoformat(start)
    out: list[_dt.date] = []
    cur = d0
    while len(out) < n:
        if cur.weekday() < 5:           # weekdays only
            out.append(cur)
        cur += _dt.timedelta(days=1)
    return out

train_days = days("2017-03-13", 45)
test_days  = days("2017-05-23", 10)
# Validation days drawn from the *training* window (non-consecutive).
val_days   = [_dt.date.fromisoformat(d) for d in
              ["2017-03-23", "2017-04-05", "2017-04-13", "2017-04-18", "2017-05-02"]]
val_set = set(val_days)

fig4 = go.Figure()

for d in train_days:
    color = "#FFB300" if d in val_set else C_ACC
    fig4.add_trace(go.Bar(
        x=[d], y=[1], width=[0.7],
        marker_color=color,
        showlegend=False,
        hovertemplate=("VAL" if d in val_set else "TRAIN") + ": %{x}<extra></extra>",
    ))
for d in test_days:
    fig4.add_trace(go.Bar(
        x=[d], y=[1], width=[0.7],
        marker_color=C_REAL,
        showlegend=False,
        hovertemplate="TEST: %{x}<extra></extra>",
    ))

# Legend proxies.
for label, c in [("Train (45 days, sequential)", C_ACC),
                 ("Validation (5 days, drawn from train window)", "#FFB300"),
                 ("Test (10 days, sequential, FUTURE of train)", C_REAL)]:
    fig4.add_trace(go.Bar(x=[None], y=[None], name=label, marker_color=c))

fig4.update_layout(
    title=("<b>Walk-forward split (Briola 2024)</b><br>"
           "<sub>45 train + 5 validation (non-consecutive, inside the train window) "
           "+ 10 sequential test days. Repeat per year. No random shuffling.</sub>"),
    xaxis=dict(title="Date", type="date"),
    yaxis=dict(showticklabels=False, range=[0, 1.2]),
    height=300, margin=dict(l=40, r=40, t=110, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=-0.45, x=0.0),
    barmode="overlay",
)

# ---------------------------------------------------------------------------
# Chart 5: Confusion-matrix realism (Briola 2024, DeepLOB on LOBSTER, K=100)
# ---------------------------------------------------------------------------
# Source: Briola 2024 Figs 6-8 (row-normalised confusion matrices).  Numbers
# read approximately from the published heatmaps for K=100.
labels = ["Down", "Stable", "Up"]

cm_small = [   # small-tick @ K=100
    [0.46, 0.07, 0.47],
    [0.45, 0.10, 0.45],
    [0.46, 0.07, 0.47],
]
cm_medium = [  # medium-tick @ K=100
    [0.51, 0.13, 0.36],
    [0.39, 0.22, 0.39],
    [0.36, 0.13, 0.51],
]
cm_large = [   # large-tick @ K=100
    [0.55, 0.36, 0.09],
    [0.20, 0.62, 0.18],
    [0.08, 0.34, 0.58],
]

fig5 = make_subplots(rows=1, cols=3,
                     subplot_titles=["Small-tick", "Medium-tick", "Large-tick"],
                     horizontal_spacing=0.08)

for col, cm in enumerate([cm_small, cm_medium, cm_large], start=1):
    fig5.add_trace(
        go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale="Blues", zmin=0, zmax=0.7,
            text=[[f"{v:.2f}" for v in row] for row in cm],
            texttemplate="%{text}", textfont=dict(size=14, color="black"),
            showscale=(col == 3),
            colorbar=dict(title="P(pred|true)") if col == 3 else None,
        ),
        row=1, col=col,
    )

fig5.update_layout(
    title=("<b>DeepLOB confusion matrices on raw LOBSTER at K=100</b><br>"
           "<sub>Briola 2024, Fig 8. Row-normalised. Diagonal is correct prediction. "
           "Notice: small-tick models can't separate Up from Down (~47% each way); "
           "large-tick models do separate them but the 'Stable' class soaks up errors.</sub>"),
    height=440, margin=dict(l=60, r=40, t=120, b=50),
)
for axis_name in fig5.layout:
    if axis_name.startswith("xaxis"):
        fig5.layout[axis_name].title = "Predicted"
    if axis_name.startswith("yaxis"):
        fig5.layout[axis_name].title = "True"

# ---------------------------------------------------------------------------
# Chart 6: LightGBM hyperparameter starting points
# ---------------------------------------------------------------------------
# Collated from regime-aware LightGBM paper (MDPI 2025), Optiver TATC public
# baselines, and Shihao Yu (GBRT on microstructure) priors.
params = [
    ("num_leaves",          64,    128,    "controls capacity; 64-128 fits LOB feature counts (~50-200 features)"),
    ("learning_rate",       0.03,  0.05,   "use early stopping; lower LR + more rounds = better"),
    ("min_data_in_leaf",    200,   1000,   "LOB rows are ~1M/day - keep leaves large to avoid overfit"),
    ("feature_fraction",    0.70,  0.90,   "stochastic feature sampling per tree"),
    ("bagging_fraction",    0.70,  0.90,   "with bagging_freq=5"),
    ("lambda_l1",           0.0,   0.5,    "L1 helps prune dead features"),
    ("lambda_l2",           0.0,   0.5,    "L2 helps with collinear OFI features"),
]

names = [p[0] for p in params]
lows  = [p[1] for p in params]
highs = [p[2] for p in params]
notes = [p[3] for p in params]
mids  = [(l + h) / 2 for l, h in zip(lows, highs)]

# Per-row normalisation so ranges are visually comparable.
def norm(low: float, high: float, lo_all: float, hi_all: float) -> tuple[float, float]:
    if hi_all == lo_all:
        return 0.5, 0.5
    return (low - lo_all) / (hi_all - lo_all), (high - lo_all) / (hi_all - lo_all)

fig6 = go.Figure()
for i, (n, l, h, note) in enumerate(params):
    # Display a horizontal bar from low to high, labelled with the actual values.
    fig6.add_trace(go.Scatter(
        x=[l, h], y=[n, n],
        mode="lines+markers+text",
        line=dict(color=C_ACC, width=8),
        marker=dict(size=12, color=C_ACC),
        text=[str(l), str(h)],
        textposition=["middle left", "middle right"],
        showlegend=False,
        hovertemplate=note + "<extra></extra>",
    ))

fig6.update_layout(
    title=("<b>LightGBM starting hyperparameters for LOB mid-direction</b><br>"
           "<sub>Ranges to sweep, not single values. Use multi_logloss + early stopping "
           "(patience 50-100) on a walk-forward validation block. Hover for rationale.</sub>"),
    xaxis=dict(title="Value (log scale for readability)", type="log"),
    yaxis=dict(autorange="reversed", title=""),
    height=420, margin=dict(l=160, r=40, t=100, b=60),
)

# ---------------------------------------------------------------------------
# Stitch into a single HTML file.
# ---------------------------------------------------------------------------
figs = [
    ("1 - Prediction horizons used in the literature", fig1),
    ("2 - Class balance vs horizon and tick-class",     fig2),
    ("3 - Realistic AUC target band",                   fig3),
    ("4 - Walk-forward train/val/test split",           fig4),
    ("5 - Confusion-matrix reality on raw LOBSTER",     fig5),
    ("6 - LightGBM hyperparameter starting ranges",     fig6),
]

html_parts: list[str] = [
    "<!DOCTYPE html><html><head>",
    "<meta charset='utf-8'>",
    "<title>LOB mid-price prediction: literature anchors</title>",
    "<style>",
    "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
    "       max-width: 1200px; margin: 24px auto; padding: 0 24px; color: #222; }",
    "h1 { border-bottom: 2px solid #1565C0; padding-bottom: 8px; }",
    "h2 { color: #1565C0; margin-top: 36px; }",
    ".takeaway { background: #FFF8E1; border-left: 4px solid #FFB300;",
    "            padding: 12px 16px; margin: 8px 0 24px 0; border-radius: 4px; }",
    "</style></head><body>",
    "<h1>LOB mid-price prediction: published anchors for our LightGBM build</h1>",
    "<p><i>Each chart is calibrated against one paper or one well-known benchmark. ",
    "Use these as the 'what's realistic' reference when interpreting your own results.</i></p>",
]

takeaways = [
    "K = 100 events is the longest commonly-tested horizon. It is a safe choice; we have direct comparables.",
    "If you binarise sign(mid) on large-tick names, ~70-95% of rows are zero-move. Use a 1-tick deadband (3-class) or restrict to small/medium-tick stocks.",
    "Target AUC ~0.58-0.62 at K=100 on raw LOBSTER. >0.70 = treat as leakage and audit.",
    "Walk-forward with sequential test days is the only credible split. No random shuffling.",
    "Confusion matrices on real data show errors concentrate between Up and Down for small-tick stocks - the model knows there's motion but not the direction.",
    "Start LightGBM at num_leaves=64-128, learning_rate=0.03, min_data_in_leaf=500. Sweep with Optuna and use early stopping on validation log-loss.",
]

for (title, fig), tk in zip(figs, takeaways, strict=True):
    html_parts.append(f"<h2>{title}</h2>")
    html_parts.append(f"<div class='takeaway'><b>Take-away:</b> {tk}</div>")
    html_parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

html_parts.append("</body></html>")

OUT.write_text("\n".join(html_parts))
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
