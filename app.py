import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# ── Sayfa Ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Likidite Analizi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
}
.metric-box {
    background: #0f1117;
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.ticker-badge {
    font-family: 'IBM Plex Mono', monospace;
    background: #1e2235;
    color: #7dd3fc;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: 600;
}
.oldest-date {
    font-size: 0.75em;
    color: #6b7280;
    font-family: 'IBM Plex Mono', monospace;
}
.pos { color: #22c55e; font-weight: 600; }
.neg { color: #ef4444; font-weight: 600; }
.neutral { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────
def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=60, show_spinner=False, persist=False)
def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = _flatten(df)
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)
    return df

@st.cache_data(ttl=30, show_spinner=False, persist=False)
def fetch_live(ticker: str) -> pd.Series | None:
    """Bugünün anlık verisini 1d periyot, 1m interval ile çek, günlük satır üret."""
    try:
        intra = yf.download(ticker, period="1d", interval="1m",
                            auto_adjust=True, progress=False)
        if intra.empty:
            return None
        intra = _flatten(intra)
        today = date.today()
        today_ts = pd.Timestamp(today)
        row = pd.Series({
            "Open":   intra["Open"].iloc[0],
            "High":   intra["High"].max(),
            "Low":    intra["Low"].min(),
            "Close":  intra["Close"].iloc[-1],
            "Volume": intra["Volume"].sum(),
        }, name=today_ts)
        return row
    except Exception:
        return None

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Günlük kapanış, güniçi değişim, Amihud, Daily Range hesapla."""
    out = pd.DataFrame(index=df.index)
    out["Kapanış (₺)"]     = df["Close"].round(2)
    out["Açılış (₺)"]      = df["Open"].round(2)
    out["Yüksek (₺)"]      = df["High"].round(2)
    out["Düşük (₺)"]       = df["Low"].round(2)
    out["Hacim"]           = df["Volume"].astype(int)

    out["Günlük Değ. (%)"] = df["Close"].pct_change() * 100
    out["Güniçi Değ. (%)"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
    out["Daily Range (₺)"] = (df["High"] - df["Low"]).round(2)
    out["Daily Range (%)"] = ((df["High"] - df["Low"]) / df["Low"] * 100).round(2)

    tl_volume = df["Close"] * df["Volume"]
    daily_return = df["Close"].pct_change().abs()
    out["Amihud (×10⁶)"] = (daily_return / tl_volume * 1e6).replace([np.inf, -np.inf], np.nan)

    out["log₁₀(Hacim)"] = np.log10(df["Volume"].replace(0, np.nan))

    amihud = out["Amihud (×10⁶)"].copy()
    log_hacim = out["log₁₀(Hacim)"].copy()
    out = out.round(4)
    out["Amihud (×10⁶)"] = amihud
    out["log₁₀(Hacim)"] = log_hacim.round(4)
    return out

def color_val(val, col):
    if pd.isna(val):
        return '<span class="neutral">—</span>'
    if col in ["Günlük Değ. (%)", "Güniçi Değ. (%)"]:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neutral")
        sign = "+" if val > 0 else ""
        return f'<span class="{cls}">{sign}{val:.2f}%</span>'
    if col == "log₁₀(Hacim)":
        return f'<span class="neutral">{val:.4f}</span>'
    if col == "Amihud (×10⁶)":
        log_val = abs(np.log10(val)) if val > 0 else np.nan
        if np.isnan(log_val):
            return '<span class="neutral">—</span>'
        return f'<span class="neutral">{log_val:.2f}</span>'
    if col == "Daily Range (₺)":
        return f'<span class="neutral">{val:.2f}</span>'
    if col == "Daily Range (%)":
        return f'<span class="neutral">{val:.2f}%</span>'
    if col == "Hacim":
        return f'<span class="neutral">{int(val):,}</span>'
    return f'<span class="neutral">{val:,.2f}</span>'

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Likidite Analizi")
    st.markdown("---")

    ticker_input = st.text_input(
        "🔍 Ticker",
        value="GARAN.IS",
        placeholder="Örn: GARAN.IS, AAPL, BTC-USD",
        help="Herhangi bir yFinance ticker girin"
    ).strip().upper()

    st.markdown("---")
    st.markdown("**📅 Başlangıç Tarihi**")
    start_date = st.date_input(
        "Başlangıç",
        value=date(1990, 1, 1),
        min_value=date(1990, 1, 1),
        max_value=date.today(),
        label_visibility="collapsed"
    )

    n_rows = st.slider("Gösterilecek Satır Sayısı", 10, 500, 60, 10)

    st.markdown("---")
    secondary_metric = st.radio(
        "📉 Grafik İkinci Eksen",
        options=["Daily Range (%)", "Amihud (×10⁶)", "Hacim"],
        index=0,
    )

    st.markdown("---")
    run = st.button("⚡ Veriyi Çek", use_container_width=True, type="primary")
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Otomatik Yenile (55s)", value=False)
    if auto_refresh:
        st.markdown(f"<span style='color:#6b7280;font-size:0.8em'>Son: {datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
        st.components.v1.html(
            "<script>setTimeout(function(){window.location.reload();}, 55000);</script>",
            height=0
        )

# ── Ana Alan ─────────────────────────────────────────────────────────────────
st.markdown("# 📈 Likidite Analizi")

if run or "last_ticker" in st.session_state:
    if run:
        st.session_state["last_ticker"] = ticker_input
        st.session_state["last_start"]  = str(start_date)

    _ticker    = st.session_state.get("last_ticker", ticker_input)
    _start     = st.session_state.get("last_start", str(start_date))
    _secondary = secondary_metric

    with st.spinner(f"{_ticker} verisi çekiliyor..."):
        raw = fetch_data(_ticker, _start)
        live = fetch_live(_ticker)

    if live is not None and not raw.empty:
        today_ts = pd.Timestamp(date.today())
        if today_ts not in raw.index:
            raw = pd.concat([raw, live.to_frame().T])
        else:
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                raw.at[today_ts, col] = live[col]

    if raw.empty:
        st.error(f"❌ {_ticker} için veri bulunamadı. Ticker'ı kontrol edin.")
    else:
        oldest = raw.index.min().strftime("%d.%m.%Y")
        newest = raw.index.max().strftime("%d.%m.%Y")
        total  = len(raw)

        col1, col2, col3, col4 = st.columns(4)
        last_close = raw["Close"].iloc[-1]
        prev_close = raw["Close"].iloc[-2] if len(raw) > 1 else last_close
        chg = ((last_close - prev_close) / prev_close) * 100
        chg_sign = "+" if chg > 0 else ""

        col1.metric("Ticker", f"{_ticker}")
        col2.metric("Son Kapanış", f"{last_close:.2f}", f"{chg_sign}{chg:.2f}%")
        col3.metric("En Eski Veri", oldest)
        col4.metric("Toplam Gün", f"{total:,}")

        st.markdown("---")

        metrics = compute_metrics(raw)
        display = metrics.iloc[::-1].head(n_rows)

        up_days   = metrics[metrics["Güniçi Değ. (%)"] > 0]
        down_days = metrics[metrics["Güniçi Değ. (%)"] < 0]
        avg_range_up_tl   = up_days["Daily Range (₺)"].mean()
        avg_range_down_tl = down_days["Daily Range (₺)"].mean()
        avg_range_up_pct  = up_days["Daily Range (%)"].mean()
        avg_range_down_pct= down_days["Daily Range (%)"].mean()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("📗 Artış Günü Ort. Range (₺)",
                   f"{avg_range_up_tl:.2f}", f"{avg_range_up_pct:.2f}%")
        sc2.metric("📗 Artış Günü Sayısı", f"{len(up_days):,}")
        sc3.metric("📕 Düşüş Günü Ort. Range (₺)",
                   f"{avg_range_down_tl:.2f}", f"{avg_range_down_pct:.2f}%",
                   delta_color="off")
        sc4.metric("📕 Düşüş Günü Sayısı", f"{len(down_days):,}")
        st.markdown("---")

        # ── Dual-Axis Grafik ─────────────────────────────────────────────────
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=metrics.index,
            y=metrics["Kapanış (₺)"],
            name="Kapanış",
            line=dict(color="#22c55e", width=1.5),
        ), secondary_y=False)

        sec_col  = _secondary
        sec_data = metrics[sec_col].dropna()

        if sec_col == "Amihud (×10⁶)":
            log_amihud = np.log10(sec_data.replace(0, np.nan).dropna()).abs()
            fig.add_trace(go.Scatter(
                x=log_amihud.index,
                y=log_amihud.values,
                name="log₁₀(Amihud)",
                line=dict(color="#f59e0b", width=1.2),
            ), secondary_y=True)
        elif sec_col == "Hacim":
            log_hacim = np.log10(metrics["Hacim"].replace(0, np.nan).dropna())
            fig.add_trace(go.Scatter(
                x=log_hacim.index,
                y=log_hacim.values,
                name="log₁₀(Hacim)",
                line=dict(color="#7dd3fc", width=1.2),
            ), secondary_y=True)
        else:
            window = min(30, len(sec_data))
            trend_vals = []
            for i in range(len(sec_data)):
                start_i = max(0, i - window + 1)
                segment = sec_data.iloc[start_i:i+1]
                x_seg   = np.arange(len(segment))
                if len(segment) >= 2:
                    z = np.polyfit(x_seg, segment.values, 1)
                    trend_vals.append(np.poly1d(z)(len(segment) - 1))
                else:
                    trend_vals.append(segment.iloc[-1])
            fig.add_trace(go.Scatter(
                x=sec_data.index,
                y=trend_vals,
                name=f"{sec_col} Trend",
                line=dict(color="#f59e0b", width=1.8),
            ), secondary_y=True)

        fig.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            legend=dict(orientation="h", y=1.05, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            dragmode="pan",
        )
        fig.update_xaxes(
            showgrid=False,
            color="#94a3b8",
            rangeslider=dict(visible=True, bgcolor="#1e2235", thickness=0.06),
            rangeselector=dict(
                bgcolor="#1e2235",
                activecolor="#7dd3fc",
                buttons=list([
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="Tümü"),
                ])
            ),
        )
        fig.update_yaxes(
            title_text="Kapanış",
            title_font=dict(color="#22c55e"),
            tickfont=dict(color="#22c55e"),
            showgrid=True,
            gridcolor="#1e2235",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="log₁₀(Amihud)" if sec_col == "Amihud (×10⁶)" else ("log₁₀(Hacim)" if sec_col == "Hacim" else sec_col),
            title_font=dict(color="#7dd3fc"),
            tickfont=dict(color="#7dd3fc"),
            showgrid=False,
            secondary_y=True,
            type="linear",
            range=[0, 6] if sec_col == "Daily Range (%)" else None,
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "modeBarButtonsToAdd": ["pan2d"], "displayModeBar": True})
        st.markdown("---")

        cols_show = [
            "Kapanış (₺)", "Açılış (₺)", "Yüksek (₺)", "Düşük (₺)",
            "Hacim", "Günlük Değ. (%)", "Güniçi Değ. (%)",
            "Daily Range (₺)", "Daily Range (%)", "Amihud (×10⁶)", "log₁₀(Hacim)"
        ]

        st.markdown(
            "<span style='font-size:0.78em;color:#6b7280;font-family:IBM Plex Mono'>"
            "ℹ️ Amihud sütunu: <b>|log₁₀(Amihud)|</b> — yüksek = daha az likit, düşük = daha likit</span>",
            unsafe_allow_html=True
        )

        header = "<tr><th>Tarih</th>" + "".join(f"<th>{c}</th>" for c in cols_show) + "</tr>"
        rows = ""
        for idx, row in display.iterrows():
            date_str = idx.strftime("%d.%m.%Y")
            cells = "".join(f"<td>{color_val(row[c], c)}</td>" for c in cols_show)
            rows += f"<tr><td><span style='font-family:IBM Plex Mono;font-size:0.85em;color:#94a3b8'>{date_str}</span></td>{cells}</tr>"

        table_html = f"""
        <style>
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82em;
            margin-top: 8px;
        }}
        .data-table th {{
            background: #1e2235;
            color: #7dd3fc;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            padding: 10px 12px;
            text-align: right;
            border-bottom: 2px solid #2a2d3e;
            white-space: nowrap;
        }}
        .data-table th:first-child {{ text-align: left; }}
        .data-table td {{
            padding: 8px 12px;
            text-align: right;
            border-bottom: 1px solid #1e2235;
        }}
        .data-table td:first-child {{ text-align: left; }}
        .data-table tr:hover td {{ background: #141824; }}
        </style>
        <div style="overflow-x:auto; max-height:65vh; overflow-y:auto;">
        <table class="data-table">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # ── İlişki Analizi ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔗 Close · Daily Range · Amihud İlişki Analizi")

        ana = pd.DataFrame({
            "Close":       metrics["Kapanış (₺)"],
            "Daily Range": metrics["Daily Range (%)"],
            "Amihud (log)": metrics["Amihud (×10⁶)"].apply(
                lambda x: abs(np.log10(x)) if pd.notna(x) and x > 0 else np.nan)
        }).dropna()

        from scipy.stats import spearmanr

        cols3 = ["Close", "Daily Range", "Amihud (log)"]
        corr_matrix = np.zeros((3, 3))
        for i, c1 in enumerate(cols3):
            for j, c2 in enumerate(cols3):
                r, _ = spearmanr(ana[c1], ana[c2])
                corr_matrix[i][j] = round(r, 3)

        heat_fig = go.Figure(go.Heatmap(
            z=corr_matrix,
            x=cols3, y=cols3,
            colorscale=[[0, "#ef4444"], [0.5, "#1e2235"], [1, "#22c55e"]],
            zmin=-1, zmax=1,
            text=corr_matrix.round(2),
            texttemplate="%{text}",
            textfont=dict(size=14, family="IBM Plex Mono"),
            showscale=True,
        ))
        heat_fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            margin=dict(l=10, r=10, t=30, b=10),
            height=280,
            title=dict(text="Spearman Korelasyon Matrisi", font=dict(color="#94a3b8", size=12)),
        )
        st.plotly_chart(heat_fig, use_container_width=True)

        st.markdown("**Rolling Spearman Korelasyon (60 gün)**")
        roll_window = 60
        pairs = [
            ("Close", "Daily Range", "#7dd3fc"),
            ("Close", "Amihud (log)", "#f59e0b"),
            ("Daily Range", "Amihud (log)", "#a78bfa"),
        ]
        roll_fig = go.Figure()
        for c1, c2, color in pairs:
            roll_corr = [
                spearmanr(ana[c1].iloc[max(0,i-roll_window):i+1],
                          ana[c2].iloc[max(0,i-roll_window):i+1])[0]
                if i >= 10 else np.nan
                for i in range(len(ana))
            ]
            roll_fig.add_trace(go.Scatter(
                x=ana.index, y=roll_corr,
                name=f"{c1} × {c2}",
                line=dict(color=color, width=1.5),
            ))
        roll_fig.add_hline(y=0, line=dict(color="#4b5563", dash="dot", width=1))
        roll_fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            yaxis=dict(range=[-1, 1], showgrid=True, gridcolor="#1e2235"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(roll_fig, use_container_width=True, config={"scrollZoom": True, "dragmode": "pan"})

        st.markdown("**Volatilite Rejimi (Daily Range medyanı bazlı)**")
        median_dr = ana["Daily Range"].median()
        ana["Rejim"] = ana["Daily Range"].apply(lambda x: "Yüksek Vol." if x >= median_dr else "Düşük Vol.")

        reg_fig = go.Figure()
        for rejim, color, dash in [("Yüksek Vol.", "#ef4444", "solid"), ("Düşük Vol.", "#22c55e", "solid")]:
            mask = ana["Rejim"] == rejim
            reg_fig.add_trace(go.Scatter(
                x=ana.index[mask], y=ana["Close"][mask],
                mode="markers",
                name=rejim,
                marker=dict(color=color, size=3, opacity=0.6),
            ))
        reg_fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            yaxis=dict(title="Kapanış", showgrid=True, gridcolor="#1e2235"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(reg_fig, use_container_width=True, config={"scrollZoom": True, "dragmode": "pan"})

        st.markdown("---")
        csv = metrics.iloc[::-1].to_csv(encoding="utf-8-sig")
        st.download_button(
            label="📥 CSV İndir (Tüm Veri)",
            data=csv,
            file_name=f"{_ticker}_{oldest.replace('.','')}_to_{newest.replace('.','')}.csv",
            mime="text/csv"
        )
else:
    st.info("👈 Soldaki konsoldan bir ticker girin ve **⚡ Veriyi Çek** butonuna tıklayın.")
    st.markdown("""
    ### Tablodaki Göstergeler

    | Gösterge | Açıklama |
    |---|---|
    | **Günlük Değ. (%)** | Önceki kapanışa göre değişim |
    | **Güniçi Değ. (%)** | (Kapanış − Açılış) / Açılış × 100 |
    | **Daily Range (₺)** | Yüksek − Düşük (mutlak fark) |
    | **Amihud (×10⁶)** | \|Getiri\| / Hacim × 10⁶ — düşük = likit |
    """)
