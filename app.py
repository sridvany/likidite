import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Ayarları
st.set_page_config(page_title="BIST Trading Terminal", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; height: 3em; }
    input { text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ BIST Profesyonel Analiz Terminali")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Veri Ayarları")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO").strip().upper()
    compare_raw = st.text_input("Korelasyon", value="XU100").strip().upper()
    
    st.divider()
    g_start = st.date_input("Başlangıç", value=datetime(2022, 1, 1))
    g_end = st.date_input("Bitiş", value=datetime.now())
    
    st.divider()
    run_analysis = st.button("ANALİZİ BAŞLAT")

def format_bist(s):
    if not s: return None
    return f"{s}.IS" if not s.endswith(".IS") else s

if run_analysis and symbol_raw:
    with st.spinner('Veriler işleniyor...'):
        t = format_bist(symbol_raw)
        ct = format_bist(compare_raw)
        
        df = yf.download(t, start=g_start, end=g_end)
        df_c = yf.download(ct, start=g_start, end=g_end)

        if not df.empty:
            # MultiIndex Temizliği
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if isinstance(df_c.columns, pd.MultiIndex): df_c.columns = df_c.columns.get_level_values(0)

            # --- HESAPLAMALAR ---
            df['Daily Range'] = (df['High'] - df['Low']).round(2)
            df['Pct'] = df['Close'].pct_change() * 100
            # Amihud Skoru (Görünür olması için çarpan yükseltildi: 10^8)
            df['Amihud'] = (df['Pct'].abs() / df['Volume'] * 100000000).round(4)

            # --- 1. ANA GRAFİK & AYRAÇ ---
            st.subheader(f"📊 {symbol_raw} Candlestick & Volume Profile")
            fig1 = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
            fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
            
            # VRP (Son 60 güne göre daha hızlı analiz)
            v_df = df.tail(100).dropna()
            v_bins = pd.cut(v_df['Close'], bins=20)
            v_prof = v_df.groupby(v_bins, observed=True)['Volume'].sum()
            fig1.add_trace(go.Bar(x=v_prof.values, y=[i.mid for i in v_prof.index], orientation='h', marker_color='rgba(255, 75, 75, 0.3)'), row=1, col=2)
            
            fig1.update_layout(height=550, template="plotly_dark", showlegend=False, xaxis=dict(rangeslider_visible=True))
            st.plotly_chart(fig1, use_container_width=True)

            # --- 2. DETAY VERİ TABLOSU ---
            st.divider()
            st.subheader("📅 Detay Veri Listesi (Son 30 Gün)")
            df_out = df.tail(30).copy()
            df_out['Değişim %'] = df_out['Pct'].apply(lambda x: f"🟢 +%{x:.2f}" if x > 0 else f"🔴 -%{abs(x):.2f}" if x < 0 else "⚪ 0.00")
            st.dataframe(df_out[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Range', 'Amihud', 'Değişim %']].sort_index(ascending=False), use_container_width=True, height=350)

            # --- 3. DUAL AXIS ANALİZLER (MODERN PLOTLY SÖZDİZİMİ) ---
            st.divider()
            
            def make_dual(y1, n1, c1, y2, n2, c2, dash=None):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=y1, name=n1, line=dict(color=c1, width=2.5), yaxis="y1"))
                fig.add_trace(go.Scatter(x=df.index, y=y2, name=n2, line=dict(color=c2, width=2.5, dash=dash), yaxis="y2"))
                
                # ValueError veren kısımlar tamamen modernize edildi
                fig.update_layout(
                    template="plotly_dark", height=500,
                    yaxis=dict(
                        title=dict(text=n1, font=dict(color=c1)),
                        tickfont=dict(color=c1),
                        autorange=True
                    ),
                    yaxis2=dict(
                        title=dict(text=n2, font=dict(color=c2)),
                        tickfont=dict(color=c2),
                        anchor="x", overlaying="y", side="right",
                        autorange=True
                    ),
                    xaxis=dict(rangeslider_visible=True), # ALTTAKİ AYRAÇLAR
                    hovermode="x unified",
                    margin=dict(t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Grafik A: Amihud vs Daily Range
            st.subheader(f"📉 {symbol_raw} - Amihud (Sol) vs Daily Range (Sağ)")
            make_dual(df['Amihud'], "Amihud (Likidite)", "#00FFCC", df['Daily Range'], "Daily Range", "#FFD700", dash='dot')
            
            # Grafik B: Daily Range vs Close
            st.subheader(f"📈 {symbol_raw} - Fiyat (Sol) vs Daily Range (Sağ)")
            make_dual(df['Close'], "Fiyat (Close)", "#FFFFFF", df['Daily Range'], "Daily Range", "#FFD700")

            # Grafik C: Amihud vs Close
            st.subheader(f"🧪 {symbol_raw} - Fiyat (Sol) vs Amihud (Sağ)")
            make_dual(df['Close'], "Fiyat (Close)", "#FFFFFF", df['Amihud'], "Amihud", "#00FFCC")

else:
    st.info("👈 Analizi başlatmak için sol menüden ayarları yapın.")
