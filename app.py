import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from scipy.stats import spearmanr          # FIX 2: dosya başına taşındı
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

@st.cache_data(ttl=3600, show_spinner=False, persist=False)
def fetch_oldest_date(ticker: str) -> str:
    # FIX 3: fast_info ile hafif sorgu; tüm tarih geçmişini çekmekten kaçınır
    try:
        t = yf.Ticker(ticker)
        epoch = t.fast_info.get("firstTradeDateEpochUtc", None)
        if epoch is not None:
            return pd.Timestamp(epoch, unit="s").strftime("%d.%m.%Y")
        # fast_info desteklemiyorsa minimal fallback
        df = yf.download(ticker, start="1990-01-01", end="1990-12-31",
                         auto_adjust=True, progress=False)
        if not df.empty:
            return df.index.min().strftime("%d.%m.%Y")
        return "—"
    except Exception:
        return "—"


@st.cache_data(ttl=120, show_spinner=False, persist=False)
def fetch_intraday(ticker: str, selected_date: str) -> pd.DataFrame:
    """Seçilen gün için 2dk veri çek."""
    try:
        df = yf.download(ticker, period="60d", interval="2m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = _flatten(df)
        df.index = df.index.tz_convert("Europe/Istanbul")
        df = df[~df.index.duplicated(keep="first")]
        df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)
        day_df = df[df.index.date == pd.Timestamp(selected_date).date()]
        return day_df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False, persist=False)
def fetch_intraday_60d(ticker: str) -> pd.DataFrame:
    """RVOL hesabı için son 60 günlük 2dk veri."""
    try:
        df = yf.download(ticker, period="60d", interval="2m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = _flatten(df)
        df.index = df.index.tz_convert("Europe/Istanbul")
        df = df[~df.index.duplicated(keep="first")]
        df.dropna(subset=["Close", "Open", "High", "Low", "Volume"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False, persist=False)
def fetch_daily_ohlc(ticker: str, selected_date: str) -> dict:
    """Seçili gün ve önceki günün 1d OHLCV verisi.
    1d seri BIST kapanış seansı (closing auction) dahil resmi değerleri içerir;
    Yahoo Finance'in gösterdiği 'previousClose' ile bire bir uyumludur.
    """
    try:
        df = yf.download(ticker, period="10d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return {}
        df = _flatten(df)
        sel = pd.Timestamp(selected_date).date()
        sel_df  = df[df.index.date == sel]
        prev_df = df[df.index.date < sel]
        out = {}
        if not sel_df.empty:
            r = sel_df.iloc[-1]
            out.update({
                "open":   float(r["Open"]),
                "high":   float(r["High"]),
                "low":    float(r["Low"]),
                "close":  float(r["Close"]),
                "volume": float(r["Volume"]),
            })
        if not prev_df.empty:
            out["prev_close"] = float(prev_df["Close"].iloc[-1])
        return out
    except Exception:
        return {}

def compute_intraday_metrics(df: pd.DataFrame, df_60d: pd.DataFrame) -> pd.DataFrame:
    """2dk bar metrikleri: fiyat, hacim, RVOL, Daily Range, Amihud, C-S Spread."""
    out = pd.DataFrame(index=df.index)
    out["Kapanış"]        = df["Close"].round(4)
    out["Açılış"]         = df["Open"].round(4)
    out["Yüksek"]         = df["High"].round(4)
    out["Düşük"]          = df["Low"].round(4)
    out["Hacim"]          = df["Volume"].astype(int)

    out["Değişim (%)"]    = df["Close"].pct_change() * 100
    out["Bar Range (%)"]  = ((df["High"] - df["Low"]) / df["Low"] * 100).round(4)

    bar_return = df["Close"].pct_change().abs()
    tl_vol     = df["Close"] * df["Volume"]
    out["Amihud (2dk)"]   = (bar_return / tl_vol * 1e6).replace([np.inf, -np.inf], np.nan)

    h = np.log(df["High"])
    l = np.log(df["Low"])
    h2 = np.log(df["High"].combine(df["High"].shift(-1), max))
    l2 = np.log(df["Low"].combine(df["Low"].shift(-1), min))
    beta  = (h - l) ** 2 + (h.shift(-1) - l.shift(-1)) ** 2
    gamma = (h2 - l2) ** 2
    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)
    cs    = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out["C-S Spread (%)"] = (cs * 100).round(4)

    # ── MEC (Market Efficiency Coefficient) — 60 bar rolling ─────────────────
    # MEC = Var(ln Ct/Ct-18) / (6 × Var(ln Ct/Ct-3))
    # Kısa: 3 bar (6dk), Uzun: 18 bar (36dk), T = 18/3 = 6
    # Random walk altında ≈ 1; ≤1 mean-revert/dayanıklı, >1 trend/yavaş döngü
    r3  = np.log(df["Close"] / df["Close"].shift(3))
    r18 = np.log(df["Close"] / df["Close"].shift(18))
    win = 60
    mec_vals = []
    for i in range(len(df)):
        if i < win:
            mec_vals.append(np.nan)
            continue
        seg3  = r3.iloc[i - win + 1: i + 1].dropna()
        seg18 = r18.iloc[i - win + 1: i + 1].dropna()
        var3  = seg3.var(ddof=1)  if len(seg3)  > 1 else np.nan
        var18 = seg18.var(ddof=1) if len(seg18) > 1 else np.nan
        if pd.notna(var18) and pd.notna(var3) and var3 > 0:
            mec_vals.append(round(var18 / (6 * var3), 4))
        else:
            mec_vals.append(np.nan)
    out["MEC"] = mec_vals

    # ── ATR (Wilder, 30 bar ≈ 1 saat) — güniçi volatilite (₺) ───────────────
    # Tek-gün df, ilk barın Cprev'i NaN → TR otomatik H−L (gap reset doğal).
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1/30, adjust=False, min_periods=30).mean().round(4)

    if not df_60d.empty:
        df_60d = df_60d.copy()
        df_60d["time_key"] = df_60d.index.strftime("%H:%M")
        avg_vol = df_60d.groupby("time_key")["Volume"].mean()
        time_keys = df.index.strftime("%H:%M")
        rvol_vals = []
        for tk, v in zip(time_keys, df["Volume"]):
            avg = avg_vol.get(tk, np.nan)
            rvol_vals.append(round(v / avg, 3) if avg and avg > 0 else np.nan)
        out["RVOL"] = rvol_vals
    else:
        out["RVOL"] = np.nan

    return out


# ── AI Yorum Yardımcıları ────────────────────────────────────────────────────
def _trend_dir(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) < 5:
        return "yetersiz"
    x = np.arange(len(s))
    try:
        slope = np.polyfit(x, s.values, 1)[0]
        rng = s.max() - s.min()
        if rng == 0 or abs(slope) < rng * 0.001:
            return "stabil"
        return "yükseliyor" if slope > 0 else "azalıyor"
    except Exception:
        return "stabil"


def _col_summary(metrics: pd.DataFrame, col: str) -> dict | None:
    if col not in metrics.columns:
        return None
    s = metrics[col].dropna()
    if s.empty:
        return None
    last_v = float(s.iloc[-1])
    last_30 = metrics[col].tail(30).dropna()
    last_90 = metrics[col].tail(90).dropna()
    last_1y = metrics[col].tail(252).dropna() if len(metrics) >= 252 else s
    pct = float((last_1y < last_v).mean() * 100) if len(last_1y) > 5 else None
    return {
        "son": round(last_v, 4),
        "30g_medyan": round(float(last_30.median()), 4) if not last_30.empty else None,
        "90g_medyan": round(float(last_90.median()), 4) if not last_90.empty else None,
        "1y_medyan": round(float(last_1y.median()), 4) if not last_1y.empty else None,
        "1y_persentil": round(pct, 1) if pct is not None else None,
        "30g_trend": _trend_dir(last_30),
    }


def build_daily_payload(metrics: pd.DataFrame, ticker: str) -> dict:
    if metrics.empty or len(metrics) < 30:
        return {}
    last = metrics.iloc[-1]
    chg30 = None
    if len(metrics) > 31:
        chg30 = round(float((metrics["Kapanış (₺)"].iloc[-1] / metrics["Kapanış (₺)"].iloc[-31] - 1) * 100), 2)
    return {
        "ticker": ticker,
        "veri_aralığı": f"{metrics.index[0].strftime('%Y-%m-%d')} → {metrics.index[-1].strftime('%Y-%m-%d')}",
        "gözlem_sayısı": int(len(metrics)),
        "fiyat": {
            "son_kapanış": round(float(last["Kapanış (₺)"]), 2),
            "son_değişim_%": round(float(last["Günlük Değ. (%)"]), 2) if pd.notna(last["Günlük Değ. (%)"]) else None,
            "30g_getiri_%": chg30,
        },
        "likidite": {
            "Daily_Range_%": _col_summary(metrics, "Daily Range (%)"),
            "Amihud_x10_6":  _col_summary(metrics, "Amihud (×10⁶)"),
            "log_Hacim":     _col_summary(metrics, "log₁₀(Hacim)"),
            "C-S_Spread_%":  _col_summary(metrics, "C-S Spread (%)"),
            "MEC":           _col_summary(metrics, "MEC"),
        },
        "volatilite": {
            "ATR_TL":        _col_summary(metrics, "ATR"),
        },
        "yön_asimetrisi_60g": _direction_asymmetry(metrics, lookback=60),
    }


def build_intraday_payload(intra: pd.DataFrame, ticker: str, sel_date: str,
                            prev_close: float | None = None) -> dict:
    if intra.empty:
        return {}
    close_p = float(intra["Kapanış"].iloc[-1])
    if prev_close is not None and prev_close > 0:
        chg_pct = (close_p / prev_close - 1) * 100
    else:
        chg_pct = (close_p / float(intra["Açılış"].iloc[0]) - 1) * 100
    return {
        "ticker": ticker,
        "tarih": sel_date,
        "bar_sayısı": int(len(intra)),
        "fiyat": {
            "açılış": round(float(intra["Açılış"].iloc[0]), 4),
            "kapanış": round(close_p, 4),
            "yüksek": round(float(intra["Yüksek"].max()), 4),
            "düşük": round(float(intra["Düşük"].min()), 4),
            "önceki_kapanış": round(float(prev_close), 4) if prev_close else None,
            "değişim_%": round(float(chg_pct), 2),
        },
        "likidite": {
            "Bar_Range_%":   _col_summary(intra, "Bar Range (%)"),
            "Amihud_2dk":    _col_summary(intra, "Amihud (2dk)"),
            "C-S_Spread_%":  _col_summary(intra, "C-S Spread (%)"),
            "RVOL":          _col_summary(intra, "RVOL"),
            "MEC":           _col_summary(intra, "MEC"),
        },
        "volatilite": {
            "ATR_TL":        _col_summary(intra, "ATR"),
        },
    }


def _direction_asymmetry(metrics: pd.DataFrame, lookback: int = 60) -> dict:
    n = min(lookback, len(metrics))
    df = metrics.tail(n)
    chg = df["Günlük Değ. (%)"]
    up   = df[chg > 0]
    down = df[chg < 0]

    def m(col, group):
        if col not in group.columns:
            return None
        s = group[col].dropna()
        return round(float(s.mean()), 4) if not s.empty else None

    return {
        "lookback_g": int(n),
        "up_gün_sayısı":   int(len(up)),
        "down_gün_sayısı": int(len(down)),
        "up_ortalama": {
            "log_Hacim":       m("log₁₀(Hacim)", up),
            "Daily_Range_%":   m("Daily Range (%)", up),
            "ATR_TL":          m("ATR", up),
            "Amihud_x10_6":    m("Amihud (×10⁶)", up),
            "C-S_Spread_%":    m("C-S Spread (%)", up),
        },
        "down_ortalama": {
            "log_Hacim":       m("log₁₀(Hacim)", down),
            "Daily_Range_%":   m("Daily Range (%)", down),
            "ATR_TL":          m("ATR", down),
            "Amihud_x10_6":    m("Amihud (×10⁶)", down),
            "C-S_Spread_%":    m("C-S Spread (%)", down),
        },
    }


def extract_top_correlations(corr_matrix, cols, top_n: int = 5) -> list:
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr_matrix[i][j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return [{"çift": f"{a} ↔ {b}", "rho": round(r, 3)} for a, b, r in pairs[:top_n]]


def build_daily_prompt(payload: dict, top_corr: list, level: str) -> str:
    import json as _json
    base = f"""Sen profesyonel bir BIST hisse senedi mikroyapı analiz uzmanısın. Aşağıdaki istatistikleri inceleyip Türkçe bir RAPOR YORUMU yaz.

VERİ:
{_json.dumps(payload, ensure_ascii=False, indent=2)}

EN GÜÇLÜ KORELASYONLAR (Spearman):
{_json.dumps(top_corr, ensure_ascii=False, indent=2)}
"""
    if level == "Az":
        rules = """
RAPOR FORMATI:
## 🎯 Genel Durum
Likidite × volatilite × fiyat üçgenini 2-3 cümleyle özetle. Rejimi adlandır (likit/illikit × volatil/sakin × dengeli/dengesiz).
"""
    elif level == "Orta":
        rules = """
RAPOR FORMATI:
## 🎯 Genel Durum
2-3 cümle özet, rejim adı.

## 💧 Likidite Profili
5 boyutu (Daily Range, Amihud, Hacim, C-S Spread, MEC) 1 yıllık persentile göre oku. Hangileri uç değerde, hangileri normal? Son 30g trendini söyle.

## 📈 Volatilite Profili
ATR seviyesini ve son 30g trendini yorumla. Yükseliyor mu, sakinleşiyor mu? Persentil bazında oku (uç/normal).

## 🔗 Yön Asimetrisi
yön_asimetrisi_60g verisinden up-day vs down-day ortalamalarını karşılaştır. Hacim ve volatilite hangi yönde baskın? Korku rejimi mi, sağlıklı ralli mi, dağıtım mı?
"""
    else:
        rules = """
RAPOR FORMATI:
## 🎯 Genel Durum
Likidite × volatilite × fiyat üçgenini 2-3 cümleyle özetle. Rejimi adlandır.

## 💧 Likidite Profili
5 boyutu (Daily Range, Amihud, Hacim, C-S Spread, MEC) 1 yıllık persentile göre oku. Her birinde son değer normal mi, uç mu? Son 30g trendini söyle. MEC'i yorumlarken 1.0 eşiğine dikkat et (≤1 dayanıklı, >1 yavaş döngü).

## 📈 Volatilite Profili
ATR seviyesini, son 30g trendini ve 1 yıllık persentilini yorumla. Yükseliyor mu, sakinleşiyor mu? Uç değerde mi, normal mi?

## 🔗 İlişki & Sinyal
En güçlü 2-3 korelasyonu yorumla. Likidite ↔ volatilite ↔ fiyat üçgeninde ne tür bir bağlanma var? Hangi metrik hangisini yönlendiriyor?

Ayrıca **yön asimetrisini** (yön_asimetrisi_60g; up-day vs down-day ortalamaları) yorumla. Aşağıdaki klasik mikroyapı kalıplarını kullan:
- Down-day vol > Up-day vol **ve** Down-day Amihud > Up-day Amihud → **KORKU REJİMİ** (satışlarda likidite kuruyor)
- Up-day log_Hacim > Down-day log_Hacim **ve** Up-day vol düşük → **SAĞLIKLI RALLİ** (kurumsal alım/birikim)
- Up-day log_Hacim < Down-day log_Hacim → **DAĞITIM SİNYALİ** (alımda hacim zayıf, satışta yüksek)
- Up-day C-S Spread < Down-day C-S Spread → likidite yükselişi destekliyor
- Up-day vol > Down-day vol **ve** Up-day Amihud düşük → **TAVAN/EUFORİ** (çıkışlarda agresif alım, fiyat tavizi yok)
Veriden hangi kalıba en yakın olduğunu söyle, sayısal farklara referans vererek.

## ⚠️ Anomali & Dikkat
Persentili %95'ten yüksek veya %5'ten düşük metrikler dikkat çekici. Son 30 gün trendinde keskin dönüş varsa belirt.
"""
    rules += """
KURALLAR:
- Türkçe yaz, sade ve teknik bir dil kullan.
- Sayısal değerlere referans vererek konuş (persentil, son değer, trend yönü).
- Veri dışı tahmin yapma; YATIRIM TAVSİYESİ VERME.
- Markdown başlıklarını (##) aynen kullan.
- Maddi/şirket-spesifik yorum yapma; sadece istatistiksel imza yorumla.
"""
    return base + rules


def build_intraday_prompt(payload: dict, level: str) -> str:
    import json as _json
    base = f"""Sen profesyonel bir BIST hisse senedi GÜNİÇİ mikroyapı uzmanısın. 2 dakikalık bar verilerinden çıkarılmış aşağıdaki istatistikleri Türkçe yorumla.

VERİ:
{_json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    if level == "Az":
        rules = "## 🎯 Gün Özeti\nGünün likidite + volatilite imzasını 2-3 cümleyle özetle.\n"
    elif level == "Orta":
        rules = """## 🎯 Gün Özeti
Günün likidite + volatilite imzasını özetle.

## 💧 Güniçi Likidite
Bar Range, Amihud, C-S Spread, RVOL, MEC profilini değerlendir. RVOL > 1.5 baskın mı, yoksa ince işlem mi? MEC ≤1 dayanıklı/mean-revert, >1 yavaş döngü.

## 📈 Güniçi Volatilite
ATR bar bazında salınım büyüklüğünü gösteriyor mu? Trend nedir?
"""
    else:
        rules = """## 🎯 Gün Özeti
Günün likidite + volatilite imzasını özetle, rejim adı ver.

## 💧 Güniçi Likidite
Bar Range, Amihud, C-S Spread, RVOL, MEC profilini değerlendir. RVOL ortalaması, persentili. Hangi metrik uç değerde? MEC'i 1.0 eşiğiyle yorumla (≤1 dayanıklı, >1 yavaş döngü).

## 📈 Güniçi Volatilite
ATR seviyesini ve seyrini yorumla. Bar başına ortalama hareket aralığı ne durumda?

## ⚠️ Anomali
Trendlerde keskin değişim, uç persentil değerleri.
"""
    rules += """
KURALLAR:
- Türkçe, sade ve teknik dil.
- Sayısal değerlere referans ver.
- Yatırım tavsiyesi verme.
- Markdown başlıkları (##) kullan.
"""
    return base + rules


def gemini_generate(api_key: str, prompt: str, max_tokens: int = 4000, temperature: float = 0.4) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    usage = resp.usage_metadata
    return {
        "text": resp.text,
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }


# Gemini-2.5-Flash fiyatlandırma (USD / 1M token, ~2025)
GEMINI_FLASH_PRICE_IN = 0.30
GEMINI_FLASH_PRICE_OUT = 2.50


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Günlük kapanış, güniçi değişim, Amihud, Daily Range, Corwin-Schultz, MEC hesapla."""
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

    # ── Corwin-Schultz Bid-Ask Spread ────────────────────────────────────────
    h = np.log(df["High"])
    l = np.log(df["Low"])
    h2 = np.log(df["High"].combine(df["High"].shift(-1), max))
    l2 = np.log(df["Low"].combine(df["Low"].shift(-1), min))

    beta  = (h - l) ** 2 + (h.shift(-1) - l.shift(-1)) ** 2
    gamma = (h2 - l2) ** 2
    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)
    cs    = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out["C-S Spread (%)"] = (cs * 100).round(4)

    # ── MEC (Market Efficiency Coefficient) — 90 günlük rolling ─────────────
    # MEC = 6 × Var(ln(Ct/Ct-5)) / Var(ln(Ct/Ct-30))
    # Kısa dönem: 5 işlem günü, Uzun dönem: 30 işlem günü, T = 30/5 = 6
    r5  = np.log(df["Close"] / df["Close"].shift(5))
    r30 = np.log(df["Close"] / df["Close"].shift(30))
    window   = 90
    mec_vals = []

    for i in range(len(df)):
        if i < window:
            mec_vals.append(np.nan)
            continue
        seg5  = r5.iloc[i - window + 1: i + 1].dropna()
        seg30 = r30.iloc[i - window + 1: i + 1].dropna()

        var5  = seg5.var(ddof=1)  if len(seg5)  > 1 else np.nan
        var30 = seg30.var(ddof=1) if len(seg30) > 1 else np.nan

        if pd.notna(var30) and var30 > 0 and pd.notna(var5):
            mec_vals.append(round(var30 / (6 * var5), 4))
        else:
            mec_vals.append(np.nan)

    out["MEC"] = mec_vals

    # ── ATR (Wilder, 14) — günlük volatilite (₺) ─────────────────────────────
    # TR = max(H−L, |H−Cprev|, |L−Cprev|) ; ATR = TR'nin Wilder ortalaması (α=1/14)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().round(2)

    amihud    = out["Amihud (×10⁶)"].copy()
    log_hacim = out["log₁₀(Hacim)"].copy()
    out = out.round(4)
    out["Amihud (×10⁶)"] = amihud
    out["log₁₀(Hacim)"]  = log_hacim.round(4)
    return out

# ── 🔬 LAB — Aşama 1: Altyapı ─────────────────────────────────────────────────
# Ham metrik tablosuna rolling z-skor (_z), rolling persentile (_pct) ve
# forward return (fwd_ret_N) sütunları ekler. Günlük ve güniçi için aynı API.

# Lab feature sütunları — scale'e göre eşleme
LAB_FEATURES_DAILY = [
    "Daily Range (%)", "Amihud (×10⁶)", "log₁₀(Hacim)",
    "C-S Spread (%)", "MEC", "ATR", "Günlük Değ. (%)",
]
LAB_FEATURES_INTRADAY = [
    "Bar Range (%)", "RVOL", "Amihud (2dk)",
    "C-S Spread (%)", "MEC", "ATR", "Değişim (%)",
]

def build_lab_frame(metrics: pd.DataFrame, *, scale: str,
                    horizons: list[int], lookback: int) -> pd.DataFrame:
    """Lab altyapı frame'i.

    Parametreler
    ------------
    metrics  : compute_metrics (daily) veya compute_intraday_metrics (intraday) çıktısı
    scale    : "daily" | "intraday"
    horizons : ufuk listesi; daily için gün, intraday için bar (örn [1,5,20] / [3,10,30])
    lookback : z-skor + persentile için rolling pencere (daily 252, intraday 60 önerilir)

    Döner
    -----
    DataFrame
        Orijinal sütunlar + her feature için "{col}_z" ve "{col}_pct" +
        her ufuk için "fwd_ret_{N}" (%). Intraday'de ek olarak "fwd_ret_eod".
    """
    if metrics is None or metrics.empty:
        return metrics

    df = metrics.copy()
    feat_cols = LAB_FEATURES_DAILY if scale == "daily" else LAB_FEATURES_INTRADAY
    feat_cols = [c for c in feat_cols if c in df.columns]

    # 1) Rolling z-skor ve persentile -----------------------------------------
    min_p = max(10, lookback // 4)
    for c in feat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        roll = s.rolling(lookback, min_periods=min_p)
        mu, sd = roll.mean(), roll.std(ddof=1)
        df[f"{c}_z"]   = ((s - mu) / sd.replace(0, np.nan)).round(4)
        try:
            df[f"{c}_pct"] = roll.rank(pct=True).round(4)
        except Exception:
            # Eski pandas için fallback
            df[f"{c}_pct"] = s.rolling(lookback, min_periods=min_p).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else np.nan,
                raw=False,
            ).round(4)

    # 2) Forward return (%) ---------------------------------------------------
    if "Kapanış" in df.columns:
        close = pd.to_numeric(df["Kapanış"], errors="coerce")
        for h in horizons:
            df[f"fwd_ret_{h}"] = ((close.shift(-h) / close - 1.0) * 100).round(4)
        if scale == "intraday" and len(close) > 1:
            last_px = close.iloc[-1]
            df["fwd_ret_eod"] = ((last_px / close - 1.0) * 100).round(4)
            df.loc[df.index[-1], "fwd_ret_eod"] = np.nan

    return df


def render_lab_panel(metrics: pd.DataFrame, *, scale: str,
                     horizons: list[int], lookback: int) -> None:
    """Aşama 1 önizleme paneli: frame özeti + son bar feature kartı + tail."""
    if metrics is None or metrics.empty:
        st.info("🔬 Lab: veri yok.")
        return

    lab = build_lab_frame(metrics, scale=scale, horizons=horizons, lookback=lookback)
    feat_cols = LAB_FEATURES_DAILY if scale == "daily" else LAB_FEATURES_INTRADAY
    feat_cols = [c for c in feat_cols if c in lab.columns]
    fwd_cols  = [c for c in lab.columns if c.startswith("fwd_ret_")]

    st.markdown("## 🔬 Lab — Aşama 1: Altyapı")
    st.caption(
        f"Ölçek: **{scale}** · Lookback: **{lookback}** · "
        f"Ufuklar: **{horizons}** · Frame: **{lab.shape[0]} × {lab.shape[1]}**"
    )

    # Son bar feature kartı: değer | z | persentil ----------------------------
    last = lab.iloc[-1]
    rows = []
    for c in feat_cols:
        rows.append({
            "Metrik": c,
            "Değer":  last[c]            if c in lab.columns else np.nan,
            "z":      last.get(f"{c}_z",   np.nan),
            "pct":    last.get(f"{c}_pct", np.nan),
        })
    card = pd.DataFrame(rows)
    st.markdown("**Son bar — feature kartı**")
    st.dataframe(
        card.style.format({"Değer": "{:.4f}", "z": "{:+.2f}", "pct": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )

    # Forward return durumu (kaç satır NaN değil) -----------------------------
    if fwd_cols:
        valid = {c: int(lab[c].notna().sum()) for c in fwd_cols}
        st.markdown("**Forward return — geçerli satır sayısı**")
        st.dataframe(
            pd.DataFrame([valid]).T.rename(columns={0: "valid_n"}),
            use_container_width=True,
        )

    # Tail preview ------------------------------------------------------------
    with st.expander("📋 Frame tail (son 20)"):
        st.dataframe(lab.tail(20), use_container_width=True)


def color_val(val, col):
    if pd.isna(val):
        return '<span class="neutral">—</span>'
    if col in ["Günlük Değ. (%)", "Güniçi Değ. (%)"]:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neutral")
        sign = "+" if val > 0 else ""
        return f'<span class="{cls}">{sign}{val:.2f}%</span>'
    if col == "C-S Spread (%)":
        return f'<span class="neutral">{val:.4f}%</span>'
    if col == "MEC":
        cls = "pos" if val <= 1.0 else "neg"
        return f'<span class="{cls}">{val:.4f}</span>'
    if col == "log₁₀(Hacim)":
        return f'<span class="neutral">{val:.4f}</span>'
    if col == "Amihud (×10⁶)":
        log_val = abs(np.log10(val)) if val > 0 else np.nan
        if np.isnan(log_val):
            return '<span class="neutral">—</span>'
        return f'<span class="neutral">{log_val:.2f}</span>'
    if col == "Daily Range (₺)":
        return f'<span class="neutral">{val:.2f}</span>'
    if col == "ATR":
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
        value="gc=f",
        placeholder="Örn: GARAN.IS, AAPL, BTC-USD",
        help="Herhangi bir yFinance ticker girin"
    ).strip().upper()

    st.markdown("---")
    analiz_modu = st.radio(
        "📐 Analiz Modu",
        options=["📅 Günlük", "📊 Güniçi"],
        index=0,
    )

    st.markdown("---")
    lab_mode = st.checkbox("🔬 Lab Modu", value=False,
                           help="Forward return + z-skor + persentile altyapısı")
    if lab_mode:
        if analiz_modu == "📅 Günlük":
            lab_horizons = st.multiselect(
                "Ufuk (gün)", options=[1, 5, 10, 20, 60],
                default=[1, 5, 20],
            )
            lab_lookback = st.slider("Lookback (gün)", 60, 504, 252, 30)
        else:
            lab_horizons = st.multiselect(
                "Ufuk (bar)", options=[3, 5, 10, 20, 30],
                default=[3, 10, 30],
            )
            lab_lookback = st.slider("Lookback (bar)", 20, 120, 60, 10)
    else:
        lab_horizons = []
        lab_lookback = 0

    st.markdown("---")
    if analiz_modu == "📅 Günlük":
        st.markdown("**📅 Başlangıç Tarihi**")
        start_date = st.date_input(
            "Başlangıç",
            value=date(2000, 1, 1),
            min_value=date(1990, 1, 1),
            max_value=date.today(),
            label_visibility="collapsed"
        )
        n_rows = st.slider("Gösterilecek Satır Sayısı", 10, 500, 60, 10)
        intraday_date = None
    else:
        st.markdown("**📅 Gün Seç (son 60 gün)**")
        intraday_date = st.date_input(
            "Gün",
            value=date.today(),
            min_value=date.today() - pd.Timedelta(days=59),
            max_value=date.today(),
            label_visibility="collapsed"
        )
        start_date = date(1990, 1, 1)
        n_rows = 60

    st.markdown("---")
    secondary_metric = st.radio(
        "📉 Likidite Boyutları",
        options=[
            "Daily Range (%) — Anındalık",
            "Amihud (×10⁶) — Genişlik",
            "Hacim — Derinlik",
            "C-S Spread (%) — Sıkılık",
            "MEC — Esneklik",
        ],
        index=0,
    )
    secondary_metric = secondary_metric.split(" — ")[0]

    volatility_metric = st.radio(
        "📈 Volatilite Boyutları",
        options=[
            "ATR — Wilder (14g / 30bar)",
        ],
        index=0,
    )
    volatility_metric = volatility_metric.split(" — ")[0]

    with st.expander("📖 Boyut Tanımları"):
        st.markdown("""
**📊 Daily Range — Anındalık**
Günlük yüksek ve düşük fiyat arasındaki mutlak fark.
Ani sipariş akışını absorbe etme kapasitesini ölçer.
Spike'lar piyasanın yeni emirleri daha az likit koşullarda karşıladığına işaret eder.

---

**📊 Amihud (2002) — Genişlik**
Günlük mutlak getirinin TL hacime oranı (×10⁶).
Bir işlem için katlanılan fiyat tavizi maliyetini temsil eder.
Yüksek = büyük fiyat etkisi = az likit.

---

**📊 Hacim — Derinlik**
Günlük toplam işlem adedi (log₁₀ normalize).
Düşük hacim zayıf likidite koşullarına işaret eder.
Yüksek = derin piyasa = büyük emirler fiyatı az etkiler.

---

**📊 Corwin-Schultz (2012) — Sıkılık**
Günlük yüksek/düşük fiyat oranından tahmin edilen bid-ask spread.
Saf işlem maliyetini temsil eder; örtülü maliyetler dahil değildir.
Düşük spread = daha iyi likidite koşulları.

---

**📊 MEC — Esneklik**
Haftalık getiri varyansının günlük getiri varyansına oranı (90 günlük rolling).
Piyasanın yeni dengesine ne kadar hızlı döndüğünü ölçer.
MEC ≈ 1 veya < 1 → piyasa dayanıklı (resilient).
MEC > 1 → fiyat yeni dengeye yavaş dönüyor = düşük esneklik.

---

**📈 ATR (Wilder, 1978) — Average True Range**
TR = max(H−L, |H−Cprev|, |L−Cprev|). ATR = TR'nin Wilder ortalaması (α = 1/N).
Günlük modda N=14, güniçi modda N=30 bar (≈1 saat).
Mutlak fiyat birimiyle (₺) bar başına ortalama hareket aralığını ölçer.
Pozisyon büyüklüğü, stop mesafesi ve volatilite rejim takibi için standart referans.
        """)

    with st.expander("📖 RVOL — Göreceli Hacim"):
        st.markdown("""
**RVOL (Relative Volume — Göreceli Hacim)**, bir zaman dilimindeki işlem hacminin, aynı zaman diliminin geçmiş ortalamasına oranıdır.

**Formül:** `RVOL = Anlık Hacim / Aynı Saatin Geçmiş Ortalama Hacmi`

**Nasıl okunur:**
- `RVOL = 1.0` → Normal gün, beklenen hacim düzeyinde
- `RVOL > 1.5` → Normalin 1.5 katı hacim — piyasada olağandışı bir hareket var
- `RVOL > 2.0` → Güçlü sinyal — kurumsal alım/satım, haber, vb.
- `RVOL < 0.8` → Zayıf hacim — piyasa ilgisiz, ince işlem
- `RVOL < 0.5` → Çok düşük katılım — büyük emirler fiyatı kolayca hareket ettirebilir

**Fiyat yönüyle birlikte okuma:**
- Fiyat ↑ + RVOL yüksek → Güçlü alım, hareket inandırıcı
- Fiyat ↑ + RVOL düşük → Zayıf alım, sürmeyebilir
- Fiyat ↓ + RVOL yüksek → Güçlü satış, panik/kurumsal çıkış
- Fiyat ↓ + RVOL düşük → Zayıf satış, teknik düzeltme olabilir

**Bu uygulamada:** Son 60 günün aynı 2dk zaman dilimine ait ortalama hacmi referans alınır.
        """)
    st.markdown("---")
    run = st.button("⚡ Veriyi Çek", use_container_width=True, type="primary")
    st.markdown("---")

    # ── AI Rapor Yorumcusu ──────────────────────────────────────────────────
    st.markdown("### 🤖 AI Rapor Yorumcusu")
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        key="gemini_api_key",
        placeholder="api anahtarı buraya...",
        help="API anahtarınız sadece bu oturumda kullanılır, hiçbir yere kaydedilmez.",
    )
    has_key = bool((gemini_key or "").strip())
    key_badge = "✅ var" if has_key else "❌ yok"
    st.markdown(
        f"<span style='color:#94a3b8;font-size:0.85em'>Model: <b>gemini-2.5-flash</b> · Key: {key_badge}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("[🔑 API key al →](https://aistudio.google.com/app/api-keys)")
    detail_level = st.select_slider(
        "Detay Seviyesi",
        options=["Az", "Orta", "Detaylı"],
        value="Detaylı",
    )
    _max_tokens_map = {"Az": 1500, "Orta": 4000, "Detaylı": 8000}
    ai_max_tokens   = _max_tokens_map[detail_level]
    ai_temperature  = 0.4
    st.markdown(
        f"<span style='color:#6b7280;font-size:0.8em'>Max token: {ai_max_tokens} · Sıcaklık: {ai_temperature}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    auto_refresh = st.checkbox("🔄 Otomatik Yenile (55s)", value=False)
    if auto_refresh:
        import pytz
        tz_tr = pytz.timezone("Europe/Istanbul")
        st.markdown(f"<span style='color:#6b7280;font-size:0.8em'>Son: {datetime.now(tz_tr).strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
        st.components.v1.html(
            "<script>setTimeout(function(){window.location.reload();}, 55000);</script>",
            height=0
        )

# ── Ana Alan ─────────────────────────────────────────────────────────────────
st.markdown("# 📈 Likidite Analizi")

if run or "last_ticker" in st.session_state:
    if run:
        st.session_state["last_ticker"]  = ticker_input
        st.session_state["last_start"]   = str(start_date)
        st.session_state["last_mode"]    = analiz_modu
        st.session_state["last_intraday_date"] = str(intraday_date) if intraday_date else None

    _ticker        = st.session_state.get("last_ticker", ticker_input)
    _start         = st.session_state.get("last_start", str(start_date))
    _secondary     = secondary_metric
    _volatility    = volatility_metric
    _mode          = st.session_state.get("last_mode", analiz_modu)
    _intraday_date = st.session_state.get("last_intraday_date")

    # ── 2 Dakikalık Mod ──────────────────────────────────────────────────────
    if _mode == "📊 Güniçi":
        sel_date = _intraday_date or str(date.today())
        with st.spinner(f"{_ticker} güniçi verisi çekiliyor..."):
            df_day  = fetch_intraday(_ticker, sel_date)
            df_60d  = fetch_intraday_60d(_ticker)

        if df_day.empty:
            st.error(f"❌ {_ticker} için {sel_date} tarihinde 2dk veri bulunamadı. Hafta sonu veya tatil olabilir.")
        else:
            st.markdown(f"### ⏱️ {_ticker} — {pd.Timestamp(sel_date).strftime('%d.%m.%Y')} Güniçi Analiz")
            intra = compute_intraday_metrics(df_day, df_60d)

            if lab_mode:
                render_lab_panel(intra, scale="intraday",
                                 horizons=lab_horizons, lookback=lab_lookback)
                st.markdown("---")

            # Resmi kapanış (auction dahil) için 1d seri; bar-level metrikler 2dk seriden
            daily = fetch_daily_ohlc(_ticker, sel_date)
            open_p  = daily.get("open",   float(df_day["Open"].iloc[0]))
            close_p = daily.get("close",  float(df_day["Close"].iloc[-1]))
            high_p  = daily.get("high",   float(df_day["High"].max()))
            low_p   = daily.get("low",    float(df_day["Low"].min()))
            prev_close = daily.get("prev_close")

            if prev_close is not None and prev_close > 0:
                gunici_chg = (close_p - prev_close) / prev_close * 100
            else:
                gunici_chg = (close_p - open_p) / open_p * 100  # fallback
            chg_sign = "+" if gunici_chg > 0 else ""

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Açılış",  f"{open_p:.2f}")
            k2.metric("Kapanış", f"{close_p:.2f}", f"{chg_sign}{gunici_chg:.2f}%")
            k3.metric("Yüksek",  f"{high_p:.2f}")
            k4.metric("Düşük",   f"{low_p:.2f}")

            # Volatilite: bugünün son bar ATR'i + 60g historical persentil etiketi
            atr_series = intra["ATR"].dropna()
            atr_today  = float(atr_series.iloc[-1]) if not atr_series.empty else None
            atr_label  = None
            if atr_today is not None and not df_60d.empty:
                def _last_atr(g):
                    pc = g["Close"].shift(1)
                    tr = pd.concat([g["High"] - g["Low"],
                                    (g["High"] - pc).abs(),
                                    (g["Low"]  - pc).abs()], axis=1).max(axis=1)
                    a = tr.ewm(alpha=1/30, adjust=False, min_periods=30).mean().dropna()
                    return a.iloc[-1] if not a.empty else np.nan
                sel_dt = pd.Timestamp(sel_date).date()
                prev_days = df_60d[df_60d.index.date != sel_dt]
                hist_atr  = (prev_days.groupby(prev_days.index.date)
                                       .apply(_last_atr).dropna())
                if len(hist_atr) >= 10:
                    p25, p75 = hist_atr.quantile(0.25), hist_atr.quantile(0.75)
                    if   atr_today > p75: atr_label = "Yüksek"
                    elif atr_today < p25: atr_label = "Düşük"
                    else:                 atr_label = "Normal"
            if atr_today is None:
                k5.metric("Volatilite (ATR)", "—")
            else:
                k5.metric("Volatilite (ATR)", f"{atr_today:.2f}",
                          atr_label, delta_color="off")
            st.markdown("---")

            def intraday_yorum(intra: pd.DataFrame, ticker: str, sel_date: str) -> None:
                if intra.empty or len(intra) < 5:
                    return

                rvol_valid = intra["RVOL"].dropna()
                if len(rvol_valid) >= 3:
                    top3_likit    = rvol_valid.nlargest(3)
                    bottom3_likit = rvol_valid.nsmallest(3)

                times = intra.index
                sabah   = intra[times.hour < 11]
                ogle    = intra[(times.hour >= 11) & (times.hour < 14)]
                kapanis = intra[times.hour >= 14]

                def dilim_rvol(d): return d["RVOL"].mean() if not d.empty and d["RVOL"].notna().any() else np.nan

                sabah_rvol   = dilim_rvol(sabah)
                ogle_rvol    = dilim_rvol(ogle)
                kapanis_rvol = dilim_rvol(kapanis)

                rvol_ort   = intra["RVOL"].mean()
                amihud_ort = intra["Amihud (2dk)"].mean()
                cs_valid   = intra[intra["C-S Spread (%)"] > 0]["C-S Spread (%)"]
                cs_ort     = cs_valid.mean() if not cs_valid.empty else np.nan
                mec_ort    = intra["MEC"].mean()

                sinyaller = {}
                if pd.notna(rvol_ort):
                    sinyaller["RVOL"]   = "iyi" if rvol_ort >= 1.2 else ("kötü" if rvol_ort < 0.8 else "nötr")
                if pd.notna(amihud_ort):
                    sinyaller["Amihud"] = "kötü" if amihud_ort > intra["Amihud (2dk)"].quantile(0.75) else "iyi"
                if pd.notna(cs_ort):
                    sinyaller["C-S"]    = "kötü" if cs_ort > cs_valid.quantile(0.75) else "iyi"
                if pd.notna(mec_ort):
                    sinyaller["MEC"]    = "iyi" if mec_ort <= 1.0 else "kötü"

                kotu = sum(1 for s in sinyaller.values() if s == "kötü")
                iyi  = sum(1 for s in sinyaller.values() if s == "iyi")
                n    = len(sinyaller)

                if n > 0:
                    if kotu >= n * 0.6:
                        genel, renk, ikon = "Düşük Likidite", "#ef4444", "🔴"
                    elif iyi >= n * 0.6:
                        genel, renk, ikon = "Yüksek Likidite", "#22c55e", "🟢"
                    else:
                        genel, renk, ikon = "Orta Likidite", "#f59e0b", "🟡"
                    st.markdown(f"### {ikon} Güniçi Likidite: <span style='color:{renk}'>{genel}</span>", unsafe_allow_html=True)

                d1, d2, d3 = st.columns(3)
                for col_w, ad, rv in [(d1, "🌅 Sabah (<11:00)", sabah_rvol),
                                       (d2, "☀️ Öğle (11-14)", ogle_rvol),
                                       (d3, "🔔 Kapanış (>14:00)", kapanis_rvol)]:
                    if pd.notna(rv):
                        r = "#22c55e" if rv >= 1.2 else ("#ef4444" if rv < 0.8 else "#f59e0b")
                        col_w.markdown(
                            f"<div style='background:#1e2235;border-left:3px solid {r};padding:10px 12px;border-radius:6px'>"
                            f"<div style='color:#94a3b8;font-size:0.75em'>{ad}</div>"
                            f"<div style='color:{r};font-weight:600'>RVOL: {rv:.2f}</div>"
                            f"</div>", unsafe_allow_html=True
                        )

                st.markdown("")
                paragraf = []
                if pd.notna(rvol_ort):
                    if rvol_ort >= 1.5:
                        paragraf.append(f"Gün genelinde ortalama RVOL **{rvol_ort:.2f}** — normalin belirgin üzerinde hacim var.")
                    elif rvol_ort < 0.8:
                        paragraf.append(f"Gün genelinde ortalama RVOL **{rvol_ort:.2f}** — ince işlem, piyasa ilgisiz.")
                    else:
                        paragraf.append(f"Gün genelinde ortalama RVOL **{rvol_ort:.2f}** — normale yakın hacim.")

                if len(rvol_valid) >= 3:
                    en_yogun  = top3_likit.index.strftime("%H:%M").tolist()
                    en_seyrek = bottom3_likit.index.strftime("%H:%M").tolist()
                    paragraf.append(f"En yoğun saatler: **{', '.join(en_yogun)}**. En seyrek saatler: **{', '.join(en_seyrek)}**.")

                dilimleri = [(sabah_rvol, "sabah"), (ogle_rvol, "öğle"), (kapanis_rvol, "kapanış")]
                gecerli = [(rv, ad) for rv, ad in dilimleri if pd.notna(rv)]
                if gecerli:
                    en_iyi  = max(gecerli, key=lambda x: x[0])
                    en_kotu = min(gecerli, key=lambda x: x[0])
                    if en_iyi[0] != en_kotu[0]:
                        paragraf.append(f"En likit dilim **{en_iyi[1]}** (RVOL: {en_iyi[0]:.2f}), en az likit dilim **{en_kotu[1]}** (RVOL: {en_kotu[0]:.2f}).")

                if pd.notna(cs_ort) and not cs_valid.empty:
                    paragraf.append(f"Ortalama C-S Spread: **%{cs_ort:.4f}** — {'işlem maliyeti yüksek' if sinyaller.get('C-S') == 'kötü' else 'işlem maliyeti normal'}.")

                st.markdown(" ".join(paragraf))

            intraday_yorum(intra, _ticker, sel_date)
            st.markdown("---")

            fig_i = make_subplots(specs=[[{"secondary_y": True}]])
            up_m   = intra["Değişim (%)"] > 0
            down_m = intra["Değişim (%)"] <= 0

            fig_i.add_trace(go.Scatter(x=intra.index, y=intra["Kapanış"],
                name="Kapanış", line=dict(color="#22c55e", width=1.5)), secondary_y=False)
            fig_i.add_trace(go.Scatter(x=intra.index[up_m], y=intra["Kapanış"][up_m],
                mode="markers", name="Artış", marker=dict(color="#22c55e", size=4),
                customdata=intra["Değişim (%)"][up_m],
                hovertemplate="%{x}<br>%{y}<br>+%{customdata:.3f}%<extra></extra>"), secondary_y=False)
            fig_i.add_trace(go.Scatter(x=intra.index[down_m], y=intra["Kapanış"][down_m],
                mode="markers", name="Düşüş", marker=dict(color="#ef4444", size=4),
                customdata=intra["Değişim (%)"][down_m],
                hovertemplate="%{x}<br>%{y}<br>%{customdata:.3f}%<extra></extra>"), secondary_y=False)
            fig_i.add_trace(go.Bar(x=intra.index, y=intra["Hacim"],
                name="Hacim", marker_color="#7dd3fc", opacity=0.3,
                visible="legendonly"), secondary_y=True)

            # ── Likidite & Volatilite boyutları (varsayılan kapalı, legend'dan aç/kapa) ──
            # Sağ eksen, açılan göstergeye göre otomatik yeniden ölçeklenir.
            atr_pct = (intra["ATR"] / intra["Kapanış"] * 100)
            log_amihud_i = np.log10(intra["Amihud (2dk)"].replace(0, np.nan)).abs()

            extra_traces = [
                ("Bar Range (%)",  intra["Bar Range (%)"], "#06b6d4"),
                ("RVOL",           intra["RVOL"],          "#ec4899"),
                ("C-S Spread (%)", intra["C-S Spread (%)"],"#a78bfa"),
                ("log₁₀|Amihud|",  log_amihud_i,           "#f59e0b"),
                ("ATR (%)",        atr_pct,                "#fb923c"),
            ]
            for name, series, color in extra_traces:
                s = series.dropna()
                fig_i.add_trace(go.Scatter(x=s.index, y=s.values,
                    name=name, line=dict(color=color, width=1.2),
                    visible="legendonly"), secondary_y=True)

            fig_i.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
                legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=40, b=10), height=380,
                title=dict(text="Güniçi Fiyat & Hacim", font=dict(color="#94a3b8", size=12)),
                dragmode="pan",
            )
            fig_i.update_xaxes(showgrid=False, color="#94a3b8")
            fig_i.update_yaxes(title_text="Kapanış", title_font=dict(color="#22c55e"),
                               tickfont=dict(color="#22c55e"), showgrid=True,
                               gridcolor="#1e2235", fixedrange=True, secondary_y=False)
            fig_i.update_yaxes(title_text="Gösterge", title_font=dict(color="#7dd3fc"),
                               tickfont=dict(color="#7dd3fc"), showgrid=False,
                               autorange=True, fixedrange=True, secondary_y=True)
            st.plotly_chart(fig_i, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            st.markdown("---")

            cols_intra = ["Kapanış", "Açılış", "Yüksek", "Düşük", "Hacim",
                          "Değişim (%)", "Bar Range (%)", "RVOL", "Amihud (2dk)", "C-S Spread (%)", "MEC", "ATR"]
            disp_intra = intra[cols_intra].iloc[::-1]

            header_i = "<tr><th>Zaman</th>" + "".join(f"<th>{c}</th>" for c in cols_intra) + "</tr>"
            rows_i = ""
            for idx, row in disp_intra.iterrows():
                zaman = idx.strftime("%H:%M")
                def cv(val, col):
                    if pd.isna(val): return '<span class="neutral">—</span>'
                    if col == "Değişim (%)":
                        cls = "pos" if val > 0 else ("neg" if val < 0 else "neutral")
                        sign = "+" if val > 0 else ""
                        return f'<span class="{cls}">{sign}{val:.3f}%</span>'
                    if col == "RVOL":
                        cls = "pos" if val >= 1.2 else ("neg" if val < 0.8 else "neutral")
                        return f'<span class="{cls}">{val:.3f}</span>'
                    if col == "Hacim":
                        return f'<span class="neutral">{int(val):,}</span>'
                    if col == "Amihud (2dk)":
                        lv = abs(np.log10(val)) if val > 0 else np.nan
                        return f'<span class="neutral">{lv:.2f}</span>' if pd.notna(lv) else '<span class="neutral">—</span>'
                    if col == "MEC":
                        cls = "pos" if val <= 1.0 else "neg"
                        return f'<span class="{cls}">{val:.4f}</span>'
                    return f'<span class="neutral">{val:.4f}</span>'
                cells = "".join(f"<td>{cv(row[c], c)}</td>" for c in cols_intra)
                rows_i += f"<tr><td><span style='font-family:IBM Plex Mono;font-size:0.85em;color:#94a3b8'>{zaman}</span></td>{cells}</tr>"

            tbl_html = f"""
            <style>
            .data-table {{ width:auto;border-collapse:collapse;font-size:0.82em;margin-top:8px;table-layout:fixed; }}
            .data-table th {{ background:#1e2235;color:#7dd3fc;font-family:'IBM Plex Mono',monospace;font-weight:600;padding:6px 4px;text-align:right;border-bottom:2px solid #2a2d3e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }}
            .data-table th:first-child {{ text-align:left;width:60px; }}
            .data-table th:not(:first-child) {{ width:90px; }}
            .data-table td {{ padding:5px 4px;text-align:right;border-bottom:1px solid #1e2235;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }}
            .data-table td:first-child {{ text-align:left; }}
            .data-table tr:hover td {{ background:#141824; }}
            </style>
            <div style="overflow-x:auto;max-height:60vh;overflow-y:auto;">
            <table class="data-table"><thead>{header_i}</thead><tbody>{rows_i}</tbody></table>
            </div>"""
            st.markdown(tbl_html, unsafe_allow_html=True)

            # ── AI Rapor Yorumcusu (güniçi) ─────────────────────────────────
            st.markdown("---")
            st.markdown("### 🤖 AI Rapor Yorumu")

            ai_c1, ai_c2 = st.columns([1, 4])
            run_ai_intra = ai_c1.button("AI Yorum Üret", use_container_width=True, type="primary",
                                        disabled=not has_key, key="run_ai_intra")
            if not has_key:
                ai_c2.info("Sidebar'dan Gemini API key girerek aktive edebilirsiniz.")

            if run_ai_intra:
                payload = build_intraday_payload(intra, _ticker, sel_date, prev_close=prev_close)
                prompt  = build_intraday_prompt(payload, detail_level)
                try:
                    with st.spinner("Gemini yorum üretiyor..."):
                        result = gemini_generate(gemini_key, prompt, ai_max_tokens, ai_temperature)
                    st.session_state["ai_report_intra"] = {
                        "ticker": _ticker, "date": sel_date, "level": detail_level, **result,
                    }
                except Exception as e:
                    st.error(f"AI çağrısı başarısız: {e}")

            rep_i = st.session_state.get("ai_report_intra")
            if rep_i and rep_i.get("ticker") == _ticker and rep_i.get("date") == sel_date:
                st.markdown(rep_i["text"])
                in_t, out_t, tot_t = rep_i["input_tokens"], rep_i["output_tokens"], rep_i["total_tokens"]
                cost = in_t * GEMINI_FLASH_PRICE_IN / 1_000_000 + out_t * GEMINI_FLASH_PRICE_OUT / 1_000_000
                st.caption(
                    f"📊 Token: **{in_t:,}** input + **{out_t:,}** output = **{tot_t:,}** toplam · "
                    f"Tahmini maliyet: **${cost:.5f}** · Detay: *{rep_i['level']}*"
                )

    # ── Günlük Mod ───────────────────────────────────────────────────────────
    else:
        with st.spinner(f"{_ticker} verisi çekiliyor..."):
            raw  = fetch_data(_ticker, _start)
            live = fetch_live(_ticker)

        if live is not None and not raw.empty:
            today_ts = pd.Timestamp(date.today())
            if today_ts not in raw.index:
                raw = pd.concat([raw, live.to_frame().T])
            else:
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    raw.at[today_ts, col] = live[col]

        # FIX 1: st.stop() ile NameError riski giderildi
        if raw.empty:
            st.error(f"❌ {_ticker} için veri bulunamadı. Ticker'ı kontrol edin.")
            st.stop()

        newest = raw.index.max().strftime("%d.%m.%Y")
        total  = len(raw)

        col1, col2, col3, col4 = st.columns(4)
        last_close = raw["Close"].iloc[-1]
        prev_close = raw["Close"].iloc[-2] if len(raw) > 1 else last_close
        chg = ((last_close - prev_close) / prev_close) * 100
        chg_sign = "+" if chg > 0 else ""

        # Tarih aralığı
        date_range = f"{raw.index.min().strftime('%m.%Y')} → {raw.index.max().strftime('%m.%Y')}"

        col1.metric("Ticker", f"{_ticker}")
        col2.metric("Son Kapanış", f"{last_close:.2f}", f"{chg_sign}{chg:.2f}%")

        # Volatilite (ATR) — son değer + 252g persentil etiketi
        metrics = compute_metrics(raw)
        display = metrics.iloc[::-1].head(n_rows)

        if lab_mode:
            render_lab_panel(metrics, scale="daily",
                             horizons=lab_horizons, lookback=lab_lookback)
            st.markdown("---")

        atr_series_d = metrics["ATR"].dropna()
        atr_today_d  = float(atr_series_d.iloc[-1]) if not atr_series_d.empty else None
        atr_label_d  = None
        if atr_today_d is not None and len(atr_series_d) > 1:
            hist = atr_series_d.iloc[-253:-1] if len(atr_series_d) > 252 else atr_series_d.iloc[:-1]
            if len(hist) >= 30:
                p25, p75 = hist.quantile(0.25), hist.quantile(0.75)
                if   atr_today_d > p75: atr_label_d = "Yüksek"
                elif atr_today_d < p25: atr_label_d = "Düşük"
                else:                   atr_label_d = "Normal"
        if atr_today_d is None:
            col3.metric("Volatilite (ATR)", "—")
        else:
            col3.metric("Volatilite (ATR)", f"{atr_today_d:.2f}",
                        atr_label_d, delta_color="off")

        col4.metric("Tarih Aralığı", date_range)

        st.markdown("---")

        def likidite_yorum(metrics: pd.DataFrame) -> None:
            m = metrics.dropna(subset=["Daily Range (%)", "Amihud (×10⁶)", "Hacim", "C-S Spread (%)", "MEC"])
            if len(m) < 21:
                st.info("Yorum için yeterli veri yok (min. 21 gün).")
                return

            son     = m.iloc[-1]
            trend_w = 20

            def pct(series, val):
                return round((series < val).mean() * 100, 1)

            dr_pct     = pct(m["Daily Range (%)"], son["Daily Range (%)"])
            amihud_v   = abs(np.log10(son["Amihud (×10⁶)"])) if son["Amihud (×10⁶)"] > 0 else np.nan
            amihud_s   = m["Amihud (×10⁶)"].apply(lambda x: abs(np.log10(x)) if x > 0 else np.nan).dropna()
            amihud_pct = pct(amihud_s, amihud_v) if amihud_v else 50
            hacim_pct  = pct(m["Hacim"], son["Hacim"])
            cs_valid   = m[m["C-S Spread (%)"] > 0]["C-S Spread (%)"]
            cs_son     = cs_valid.iloc[-1] if len(cs_valid) > 0 else None
            cs_pct     = pct(cs_valid, cs_son) if cs_son is not None else None
            mec_pct    = pct(m["MEC"].dropna(), son["MEC"]) if pd.notna(son["MEC"]) else None

            def trend(series):
                s = series.dropna()
                if len(s) < trend_w * 2:
                    return 0
                return s.iloc[-trend_w:].mean() - s.iloc[-trend_w*2:-trend_w].mean()

            dr_trend     = trend(m["Daily Range (%)"])
            amihud_trend = trend(amihud_s)
            hacim_trend  = trend(m["Hacim"])
            cs_trend     = trend(cs_valid) if len(cs_valid) >= trend_w * 2 else 0
            mec_trend    = trend(m["MEC"].dropna())

            sinyaller = {}

            def sinyal_ters(pct_val, tr):
                s = "kötü" if pct_val >= 75 else ("nötr" if pct_val >= 50 else "iyi")
                t = "↑" if tr > 0 else ("↓" if tr < 0 else "→")
                return s, t

            def sinyal_duz(pct_val, tr):
                s = "iyi" if pct_val >= 75 else ("nötr" if pct_val >= 50 else "kötü")
                t = "↑" if tr > 0 else ("↓" if tr < 0 else "→")
                return s, t

            sinyaller["Daily Range"] = sinyal_ters(dr_pct,     dr_trend)
            sinyaller["Amihud"]      = sinyal_ters(amihud_pct, amihud_trend)
            sinyaller["Hacim"]       = sinyal_duz (hacim_pct,  hacim_trend)
            if cs_pct is not None:
                sinyaller["C-S Spread"] = sinyal_ters(cs_pct, cs_trend)
            if mec_pct is not None:
                mec_sinyal   = "kötü" if son["MEC"] > 1 else ("nötr" if son["MEC"] > 0.8 else "iyi")
                mec_trend_ok = "↑" if mec_trend > 0 else ("↓" if mec_trend < 0 else "→")
                sinyaller["MEC"] = (mec_sinyal, mec_trend_ok)

            kotu   = sum(1 for s, _ in sinyaller.values() if s == "kötü")
            iyi    = sum(1 for s, _ in sinyaller.values() if s == "iyi")
            toplam = len(sinyaller)

            if kotu >= toplam * 0.6:
                genel, renk, ikon = "Düşük Likidite", "#ef4444", "🔴"
            elif iyi >= toplam * 0.6:
                genel, renk, ikon = "Yüksek Likidite", "#22c55e", "🟢"
            else:
                genel, renk, ikon = "Orta Likidite", "#f59e0b", "🟡"

            renk_map   = {"iyi": "#22c55e", "nötr": "#94a3b8", "kötü": "#ef4444"}
            etiket_map = {
                "Daily Range": ("Anındalık", f"%{dr_pct:.0f} persentil"),
                "Amihud":      ("Genişlik",  f"%{amihud_pct:.0f} persentil"),
                "Hacim":       ("Derinlik",  f"%{hacim_pct:.0f} persentil"),
                "C-S Spread":  ("Sıkılık",   f"%{cs_pct:.0f} persentil" if cs_pct is not None else "—"),
                "MEC":         ("Esneklik",  f"MEC = {son['MEC']:.3f}" if pd.notna(son["MEC"]) else "—"),
            }

            st.markdown(f"### {ikon} Likidite Durumu: <span style='color:{renk}'>{genel}</span>", unsafe_allow_html=True)

            boyut_cols = st.columns(len(sinyaller))
            for col_i, (boyut, (sinyal, trend_ok)) in enumerate(sinyaller.items()):
                r = renk_map[sinyal]
                ad, detay = etiket_map[boyut]
                boyut_cols[col_i].markdown(
                    f"<div style='background:#1e2235;border-left:3px solid {r};padding:10px 12px;border-radius:6px'>"
                    f"<div style='color:#94a3b8;font-size:0.75em;font-family:IBM Plex Mono'>{boyut}</div>"
                    f"<div style='color:{r};font-weight:600;font-size:1em'>{ad}</div>"
                    f"<div style='color:#94a3b8;font-size:0.78em'>{detay} {trend_ok}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("")
            paragraf = []

            if genel == "Düşük Likidite":
                paragraf.append(f"**{_ticker}** bugün itibarıyla **düşük likidite** koşullarında işlem görüyor.")
            elif genel == "Yüksek Likidite":
                paragraf.append(f"**{_ticker}** bugün itibarıyla **yüksek likidite** koşullarında işlem görüyor.")
            else:
                paragraf.append(f"**{_ticker}** bugün itibarıyla **karma likidite** koşullarında işlem görüyor.")

            if sinyaller.get("Amihud", ("nötr",))[0] == "kötü" and sinyaller.get("Hacim", ("nötr",))[0] == "kötü":
                paragraf.append("Fiyat etkisi yüksek, işlem hacmi düşük — büyük emirler ciddi kayma yaratabilir.")
            elif sinyaller.get("Amihud", ("nötr",))[0] == "iyi" and sinyaller.get("Hacim", ("nötr",))[0] == "iyi":
                paragraf.append("Fiyat etkisi düşük ve hacim güçlü — emir gerçekleştirme maliyeti tarihsel olarak düşük.")
            elif sinyaller.get("Amihud", ("nötr",))[0] != sinyaller.get("Hacim", ("nötr",))[0]:
                paragraf.append("Amihud ve hacim sinyalleri çelişiyor — likidite konusunda temkinli olmak gerekir.")

            if sinyaller.get("Daily Range", ("nötr",))[0] == "kötü":
                paragraf.append(f"Günlük fiyat aralığı tarihsel dağılımın %{dr_pct:.0f}'lik dilimine girmiş; anındalık zayıf.")

            if cs_pct is not None and sinyaller.get("C-S Spread", ("nötr",))[0] == "kötü":
                paragraf.append(f"Bid-ask spread (son geçerli: %{cs_son:.4f}) %{cs_pct:.0f} persentilde — işlem maliyeti yüksek.")

            if pd.notna(son.get("MEC")):
                if son["MEC"] > 1:
                    paragraf.append(f"MEC = {son['MEC']:.3f} (>1): Fiyat yeni dengesine yavaş dönüyor, piyasa esnekliği zayıf.")
                else:
                    paragraf.append(f"MEC = {son['MEC']:.3f} (≤1): Fiyat yeni dengesine hızlı dönüyor, piyasa dayanıklı.")

            kotu_trend = sum(1 for _, t in sinyaller.values() if t == "↑" and _ == "kötü")
            iyi_trend  = sum(1 for _, t in sinyaller.values() if t == "↓" and _ == "kötü")
            if kotu_trend >= 2:
                paragraf.append("Son 20 günlük trend likiditenin **kötüleştiğine** işaret ediyor.")
            elif iyi_trend >= 2:
                paragraf.append("Son 20 günlük trend likiditenin **iyileştiğine** işaret ediyor.")

            st.markdown(" ".join(paragraf))

        likidite_yorum(metrics)
        st.markdown("---")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=metrics.index, y=metrics["Kapanış (₺)"],
            name="Kapanış", line=dict(color="#22c55e", width=1.5),
        ), secondary_y=False)

        up_mask   = metrics["Günlük Değ. (%)"] > 0
        down_mask = metrics["Günlük Değ. (%)"] < 0
        fig.add_trace(go.Scatter(
            x=metrics.index[up_mask], y=metrics["Kapanış (₺)"][up_mask],
            mode="markers", name="Artış Günü",
            marker=dict(color="#22c55e", size=4, symbol="circle"),
            hovertemplate="%{x}<br>Kapanış: %{y}<br>Değ: +%{customdata:.2f}%<extra></extra>",
            customdata=metrics["Günlük Değ. (%)"][up_mask],
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=metrics.index[down_mask], y=metrics["Kapanış (₺)"][down_mask],
            mode="markers", name="Düşüş Günü",
            marker=dict(color="#ef4444", size=4, symbol="circle"),
            hovertemplate="%{x}<br>Kapanış: %{y}<br>Değ: %{customdata:.2f}%<extra></extra>",
            customdata=metrics["Günlük Değ. (%)"][down_mask],
        ), secondary_y=False)

        sec_col  = _secondary
        sec_data = metrics[sec_col].dropna()

        if sec_col == "Amihud (×10⁶)":
            log_amihud = np.log10(sec_data.replace(0, np.nan).dropna()).abs()
            fig.add_trace(go.Scatter(x=log_amihud.index, y=log_amihud.values,
                name="log₁₀(Amihud)", line=dict(color="#f59e0b", width=1.2)), secondary_y=True)
        elif sec_col == "Hacim":
            log_hacim = np.log10(metrics["Hacim"].replace(0, np.nan).dropna())
            fig.add_trace(go.Scatter(x=log_hacim.index, y=log_hacim.values,
                name="log₁₀(Hacim)", line=dict(color="#7dd3fc", width=1.2)), secondary_y=True)
        elif sec_col == "C-S Spread (%)":
            fig.add_trace(go.Scatter(x=sec_data.index, y=sec_data.values,
                name="C-S Spread (%)", line=dict(color="#a78bfa", width=1.2)), secondary_y=True)
        elif sec_col == "MEC":
            fig.add_trace(go.Scatter(x=sec_data.index, y=sec_data.values,
                name="MEC", line=dict(color="#fb923c", width=1.2)), secondary_y=True)
            fig.add_hline(y=1.0, line=dict(color="#6b7280", dash="dot", width=1), secondary_y=True)
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
            fig.add_trace(go.Scatter(x=sec_data.index, y=trend_vals,
                name=f"{sec_col} Trend", line=dict(color="#f59e0b", width=1.8)), secondary_y=True)

        # ── Volatilite trace (sağ eksen, ikinci çizgi) ──────────────────────
        vol_col   = _volatility
        # ATR grafikte yüzde olarak gösterilir (ATR/Close*100) — sağ eksenle (Daily Range %) tutarlı
        if vol_col == "ATR":
            vol_series_pct = (metrics["ATR"] / metrics["Kapanış (₺)"] * 100).dropna()
            vol_data  = vol_series_pct
            vol_label = "ATR (%)"
        else:
            vol_data  = metrics[vol_col].dropna()
            vol_label = vol_col
        vol_color = "#f59e0b"  # ATR — turuncu
        fig.add_trace(go.Scatter(x=vol_data.index, y=vol_data.values,
            name=vol_label, line=dict(color=vol_color, width=1.2, dash="dot")), secondary_y=True)

        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            margin=dict(l=10, r=10, t=40, b=10), height=400, dragmode="pan",
        )
        fig.update_xaxes(
            showgrid=False, color="#94a3b8",
            rangeslider=dict(visible=True, bgcolor="#1e2235", thickness=0.06),
            rangeselector=dict(
                bgcolor="#1e2235", activecolor="#7dd3fc",
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
        fig.update_yaxes(title_text="Kapanış", title_font=dict(color="#22c55e"),
                         tickfont=dict(color="#22c55e"), showgrid=True,
                         gridcolor="#1e2235", secondary_y=False)
        fig.update_yaxes(
            title_text=(
                "log₁₀(Amihud)" if sec_col == "Amihud (×10⁶)" else
                "log₁₀(Hacim)"  if sec_col == "Hacim" else
                "C-S Spread (%)" if sec_col == "C-S Spread (%)" else
                "MEC"            if sec_col == "MEC" else sec_col
            ),
            title_font=dict(color="#7dd3fc"), tickfont=dict(color="#7dd3fc"),
            showgrid=False, secondary_y=True, type="linear",
            range=[0, 6] if sec_col == "Daily Range (%)" else None,
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "modeBarButtonsToAdd": ["pan2d"], "displayModeBar": True})
        st.markdown("---")

        cols_show = [
            "Günlük Değ. (%)", "Güniçi Değ. (%)",
            "Daily Range (₺)", "Daily Range (%)", "Amihud (×10⁶)", "log₁₀(Hacim)",
            "C-S Spread (%)", "MEC", "ATR"
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
        .data-table {{ width:auto;border-collapse:collapse;font-size:0.82em;margin-top:8px;table-layout:fixed; }}
        .data-table th {{ background:#1e2235;color:#7dd3fc;font-family:'IBM Plex Mono',monospace;font-weight:600;padding:6px 4px;text-align:right;border-bottom:2px solid #2a2d3e;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }}
        .data-table th:first-child {{ text-align:left;width:80px; }}
        .data-table th:not(:first-child) {{ width:90px; }}
        .data-table td {{ padding:5px 4px;text-align:right;border-bottom:1px solid #1e2235;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }}
        .data-table td:first-child {{ text-align:left; }}
        .data-table tr:hover td {{ background:#141824; }}
        </style>
        <div style="overflow-x:auto;max-height:65vh;overflow-y:auto;">
        <table class="data-table"><thead>{header}</thead><tbody>{rows}</tbody></table>
        </div>"""
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔗 Likidite & Volatilite Boyutları İlişki Analizi")

        ana = pd.DataFrame({
            "Close":        metrics["Kapanış (₺)"],
            "Daily Range":  metrics["Daily Range (%)"],
            "Amihud (log)": metrics["Amihud (×10⁶)"].apply(
                lambda x: abs(np.log10(x)) if pd.notna(x) and x > 0 else np.nan),
            "Hacim (log)":  metrics["log₁₀(Hacim)"],
            "C-S Spread":   metrics["C-S Spread (%)"],
            "MEC":          metrics["MEC"],
            "ATR":          metrics["ATR"],
        }).dropna()

        cols3 = ["Close", "Daily Range", "Amihud (log)", "Hacim (log)", "C-S Spread", "MEC", "ATR"]
        n = len(cols3)
        corr_matrix = np.zeros((n, n))
        for i, c1 in enumerate(cols3):
            for j, c2 in enumerate(cols3):
                r, _ = spearmanr(ana[c1], ana[c2])
                corr_matrix[i][j] = round(r, 3)

        heat_fig = go.Figure(go.Heatmap(
            z=corr_matrix, x=cols3, y=cols3,
            colorscale=[[0, "#ef4444"], [0.5, "#1e2235"], [1, "#22c55e"]],
            zmin=-1, zmax=1,
            text=corr_matrix.round(2), texttemplate="%{text}",
            textfont=dict(size=12, family="IBM Plex Mono"), showscale=True,
        ))
        heat_fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            margin=dict(l=10, r=10, t=30, b=10), height=380,
            title=dict(text="Spearman Korelasyon Matrisi", font=dict(color="#94a3b8", size=12)),
        )
        st.plotly_chart(heat_fig, use_container_width=True)

        st.markdown("**Rolling Spearman Korelasyon (60 gün)**")
        roll_window = 60
        pairs = [
            ("Close", "Daily Range",  "#7dd3fc"),
            ("Close", "Amihud (log)", "#f59e0b"),
            ("Close", "Hacim (log)",  "#22c55e"),
            ("Close", "C-S Spread",   "#a78bfa"),
            ("Close", "MEC",          "#fb923c"),
            ("Close", "ATR",          "#f59e0b"),
        ]
        roll_fig = go.Figure()
        for c1, c2, color in pairs:
            roll_corr = [
                spearmanr(ana[c1].iloc[max(0,i-roll_window):i+1],
                          ana[c2].iloc[max(0,i-roll_window):i+1])[0]
                if i >= 10 else np.nan
                for i in range(len(ana))
            ]
            roll_fig.add_trace(go.Scatter(x=ana.index, y=roll_corr,
                name=f"{c1} × {c2}", line=dict(color=color, width=1.5)))
        roll_fig.add_hline(y=0, line=dict(color="#4b5563", dash="dot", width=1))
        roll_fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=30, b=10), height=300,
            yaxis=dict(range=[-1, 1], showgrid=True, gridcolor="#1e2235"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(roll_fig, use_container_width=True, config={"scrollZoom": True, "dragmode": "pan"})

        # ── AI Rapor Yorumcusu (günlük) ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 AI Rapor Yorumu")

        c1, c2 = st.columns([1, 4])
        run_ai = c1.button("AI Yorum Üret", use_container_width=True, type="primary",
                           disabled=not has_key, key="run_ai_daily")
        if not has_key:
            c2.info("Sidebar'dan Gemini API key girerek aktive edebilirsiniz.")

        if run_ai:
            payload   = build_daily_payload(metrics, _ticker)
            top_corr  = extract_top_correlations(corr_matrix, cols3, top_n=6)
            prompt    = build_daily_prompt(payload, top_corr, detail_level)
            try:
                with st.spinner("Gemini yorum üretiyor..."):
                    result = gemini_generate(gemini_key, prompt, ai_max_tokens, ai_temperature)
                st.session_state["ai_report_daily"] = {
                    "ticker": _ticker, "level": detail_level, **result,
                }
            except Exception as e:
                st.error(f"AI çağrısı başarısız: {e}")

        rep = st.session_state.get("ai_report_daily")
        if rep and rep.get("ticker") == _ticker:
            st.markdown(rep["text"])
            in_t, out_t, tot_t = rep["input_tokens"], rep["output_tokens"], rep["total_tokens"]
            cost = in_t * GEMINI_FLASH_PRICE_IN / 1_000_000 + out_t * GEMINI_FLASH_PRICE_OUT / 1_000_000
            st.caption(
                f"📊 Token: **{in_t:,}** input + **{out_t:,}** output = **{tot_t:,}** toplam · "
                f"Tahmini maliyet: **${cost:.5f}** · Detay: *{rep['level']}*"
            )

        st.markdown("---")
        import io
        excel_df = metrics.iloc[::-1].copy()
        excel_df.index.name = "Date"
        excel_df = excel_df.reset_index()
        excel_df["Date"] = excel_df["Date"].dt.strftime("%d.%m.%Y")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            excel_df.to_excel(writer, index=False, sheet_name=_ticker)
        st.download_button(
            label="📥 Excel İndir (Tüm Veri)",
            data=buf.getvalue(),
            file_name=f"{_ticker}_{newest.replace('.','')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👈 Soldaki konsoldan bir ticker girin ve **⚡ Veriyi Çek** butonuna tıklayın.")
    st.markdown("""
    ### Tablodaki Göstergeler

    | Gösterge | Açıklama |
    |---|---|
    | **Günlük Değ. (%)** | Önceki kapanışa göre değişim |
    | **Güniçi Değ. (%)** | (Kapanış − Önceki Kapanış) / Önceki Kapanış × 100 — yfinance standardı |
    | **Daily Range (₺)** | Yüksek − Düşük (mutlak fark) |
    | **Amihud (×10⁶)** | |Getiri| / Hacim × 10⁶ — düşük = likit |
    """)
