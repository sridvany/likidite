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
    page_title="BIST30 Analiz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Şirket → Ticker Haritası ─────────────────────────────────────────────────
TICKER_MAP = {
    "Anadolu Efes": "AEFES.IS",
    "Akbank": "AKBNK.IS",
    "Aselsan": "ASELS.IS",
    "BIM": "BIMAS.IS",
    "Emlak Konut": "EKGYO.IS",
    "ENKA": "ENKAI.IS",
    "Erdemir": "EREGL.IS",
    "Ford Otosan": "FROTO.IS",
    "Garanti": "GARAN.IS",
    "Gübretaş": "GUBRF.IS",
    "İş Bankası": "ISCTR.IS",
    "Koç Holding": "KCHOL.IS",
    "Koza Altın": "KOZAL.IS",
    "Kardemir": "KRDMD.IS",
    "Migros": "MGROS.IS",
    "Petkim": "PETKM.IS",
    "Sabancı Holding": "SAHOL.IS",
    "SASA": "SASA.IS",
    "Şişecam": "SISE.IS",
    "TAV": "TAVHL.IS",
    "Turkcell": "TCELL.IS",
    "THY": "THYAO.IS",
    "Tofaş": "TOASO.IS",
    "Türk Telekom": "TTKOM.IS",
    "Tüpraş": "TUPRS.IS",
    "Vakıfbank": "VAKBN.IS",
    "Yapı Kredi": "YKBNK.IS",
    "Pegasus": "PGSUS.IS",
    "Astor": "ASTOR.IS",
    "Destek Finans Faktoring": "DSTKF.IS",
}

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

    # Günlük Kapanış Değişimi (önceki kapanışa göre %)
    out["Günlük Değ. (%)"] = df["Close"].pct_change() * 100

    # Güniçi Değişim: (Kapanış - Açılış) / Açılış * 100
    out["Güniçi Değ. (%)"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100

    # Daily Range: High - Low (mutlak fark)
    out["Daily Range (₺)"] = (df["High"] - df["Low"]).round(2)
    # Daily Range %: (High - Low) / Low * 100
    out["Daily Range (%)"] = ((df["High"] - df["Low"]) / df["Low"] * 100).round(2)

    # Amihud İlliquidity: |Return| / TL Hacim  (×10^6 ölçeklendi)
    tl_volume = df["Close"] * df["Volume"]
    daily_return = df["Close"].pct_change().abs()
    out["Amihud (×10⁶)"] = (daily_return / tl_volume * 1e6).replace([np.inf, -np.inf], np.nan)

    # Amihud hariç diğerlerini yuvarla, Amihud tam hassasiyette kalsın
    amihud = out["Amihud (×10⁶)"].copy()
    out = out.round(4)
    out["Amihud (×10⁶)"] = amihud
    return out

def color_val(val, col):
    if pd.isna(val):
        return '<span class="neutral">—</span>'
    if col in ["Günlük Değ. (%)", "Güniçi Değ. (%)"]:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neutral")
        sign = "+" if val > 0 else ""
        return f'<span class="{cls}">{sign}{val:.2f}%</span>'
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
    st.markdown("## 📊 BIST30 Konsol")
    st.markdown("---")

    company_names = sorted(TICKER_MAP.keys())
    selected_company = st.selectbox(
        "🔍 Şirket Seç",
        options=company_names,
        index=company_names.index("Garanti"),
        help="Şirket adını yazarak filtreleyebilirsiniz"
    )

    ticker = TICKER_MAP[selected_company]
    st.markdown(f"**Ticker:** <span class='ticker-badge'>{ticker}</span>", unsafe_allow_html=True)

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
        options=["Daily Range (%)", "Amihud (×10⁶)"],
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
st.markdown("# 📈 BIST30 Günlük Analiz Tablosu")

if run or "last_ticker" in st.session_state:
    if run:
        st.session_state["last_ticker"] = ticker
        st.session_state["last_start"]  = str(start_date)
        st.session_state["last_company"] = selected_company

    _ticker  = st.session_state.get("last_ticker", ticker)
    _start   = st.session_state.get("last_start", str(start_date))
    _company = st.session_state.get("last_company", selected_company)
    _secondary = secondary_metric

    with st.spinner(f"{_company} verisi çekiliyor..."):
        raw = fetch_data(_ticker, _start)
        live = fetch_live(_ticker)

    # Anlık satırı birleştir (bugünün kapanışı henüz yoksa ekle)
    if live is not None and not raw.empty:
        today_ts = pd.Timestamp(date.today())
        if today_ts not in raw.index:
            raw = pd.concat([raw, live.to_frame().T])
        else:
            # Güniçi verilerle güncelle
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                raw.at[today_ts, col] = live[col]

    if raw.empty:
        st.error(f"❌ {_ticker} için veri bulunamadı.")
    else:
        oldest = raw.index.min().strftime("%d.%m.%Y")
        newest = raw.index.max().strftime("%d.%m.%Y")
        total  = len(raw)

        # Üst bilgi kartları
        col1, col2, col3, col4 = st.columns(4)
        last_close = raw["Close"].iloc[-1]
        prev_close = raw["Close"].iloc[-2] if len(raw) > 1 else last_close
        chg = ((last_close - prev_close) / prev_close) * 100
        chg_sign = "+" if chg > 0 else ""

        col1.metric("Şirket", f"{_company}", f"{_ticker}")
        col2.metric("Son Kapanış", f"₺{last_close:.2f}", f"{chg_sign}{chg:.2f}%")
        col3.metric("En Eski Veri", oldest)
        col4.metric("Toplam Gün", f"{total:,}")

        st.markdown("---")

        # Metrik tablosu
        metrics = compute_metrics(raw)
        display = metrics.iloc[::-1].head(n_rows)  # En yeniden eskiye

        # ── Özet: Artış / Düşüş günleri Daily Range ortalaması ──────────────
        up_days   = metrics[metrics["Güniçi Değ. (%)"] > 0]
        down_days = metrics[metrics["Güniçi Değ. (%)"] < 0]
        avg_range_up_tl   = up_days["Daily Range (₺)"].mean()
        avg_range_down_tl = down_days["Daily Range (₺)"].mean()
        avg_range_up_pct  = up_days["Daily Range (%)"].mean()
        avg_range_down_pct= down_days["Daily Range (%)"].mean()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("📗 Artış Günü Ort. Range (₺)",
                   f"₺{avg_range_up_tl:.2f}", f"{avg_range_up_pct:.2f}%")
        sc2.metric("📗 Artış Günü Sayısı", f"{len(up_days):,}")
        sc3.metric("📕 Düşüş Günü Ort. Range (₺)",
                   f"₺{avg_range_down_tl:.2f}", f"{avg_range_down_pct:.2f}%",
                   delta_color="off")
        sc4.metric("📕 Düşüş Günü Sayısı", f"{len(down_days):,}")
        st.markdown("---")

        # ── Dual-Axis Grafik ─────────────────────────────────────────────────
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=metrics.index,
            y=metrics["Kapanış (₺)"],
            name="Kapanış (₺)",
            line=dict(color="#22c55e", width=1.5),
        ), secondary_y=False)

        # İkinci eksen verisi
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
            title_text="Kapanış (₺)",
            title_font=dict(color="#22c55e"),
            tickfont=dict(color="#22c55e"),
            showgrid=True,
            gridcolor="#1e2235",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="log₁₀(Amihud)" if sec_col == "Amihud (×10⁶)" else sec_col,
            title_font=dict(color="#7dd3fc"),
            tickfont=dict(color="#7dd3fc"),
            showgrid=False,
            secondary_y=True,
            type="linear",
            range=[0, 6] if sec_col == "Daily Range (%)" else None,
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "modeBarButtonsToAdd": ["pan2d"], "displayModeBar": True})
        st.markdown("---")

        # HTML tablo oluştur
        cols_show = [
            "Kapanış (₺)", "Açılış (₺)", "Yüksek (₺)", "Düşük (₺)",
            "Hacim", "Günlük Değ. (%)", "Güniçi Değ. (%)",
            "Daily Range (₺)", "Daily Range (%)", "Amihud (×10⁶)"
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

        # İndir butonu
        st.markdown("---")
        csv = metrics.iloc[::-1].to_csv(encoding="utf-8-sig")
        st.download_button(
            label="📥 CSV İndir (Tüm Veri)",
            data=csv,
            file_name=f"{_ticker}_{oldest.replace('.','')}_to_{newest.replace('.','')}.csv",
            mime="text/csv"
        )
else:
    st.info("👈 Soldaki konsoldan bir şirket seçip **⚡ Veriyi Çek** butonuna tıklayın.")
    st.markdown("""
    ### Tablodaki Göstergeler

    | Gösterge | Açıklama |
    |---|---|
    | **Günlük Değ. (%)** | Önceki kapanışa göre değişim |
    | **Güniçi Değ. (%)** | (Kapanış − Açılış) / Açılış × 100 |
    | **Daily Range (₺)** | Yüksek − Düşük (TL cinsinden mutlak fark) |
    | **Amihud (×10⁶)** | \|Getiri\| / TL Hacim × 10⁶ — düşük = likit |
    """)
