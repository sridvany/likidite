import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Ayarları
st.set_page_config(page_title="BIST Analiz Terminali", layout="wide")

# Görsel Stil Düzenlemeleri
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; height: 3em; }
    input { text-transform: uppercase; }
    /* Slider (Ayraç) stilini belirginleştir */
    .stSlider { padding-bottom: 3rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ BIST Profesyonel Analiz & Zaman Tüneli")

# --- SIDEBAR (Global Veri Çekme) ---
with st.sidebar:
    st.header("🔍 Global Veri Çek")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO").strip().upper()
    compare_raw = st.text_input("Korelasyon Sembolü", value="XU100").strip().upper()
    
    st.divider()
    st.subheader("📅 Ana Veri Aralığı")
    # Çok geniş bir aralık seçilebilir (1990 - Bugün)
    min_date = datetime(1990, 1, 1)
    max_date = datetime.now()
    default_start = datetime.now() - timedelta(days=365*2) # Varsayılan son 2 yıl
    
    g_start = st.date_input("Global Başlangıç", value=default_start, min_value=min_date, max_value=max_date)
    g_end = st.date_input("Global Bitiş", value=max_date, min_value=min_date, max_value=max_date)
    
    st.divider()
    run_analysis = st.button("VERİLERİ GETİR VE ANALİZ ET")

def format_bist(s):
    if not s: return None
    return f"{s}.IS" if not s.endswith(".IS") else s

if run_analysis:
    if not symbol_raw:
        st.warning("⚠️ Lütfen bir sembol giriniz.")
    else:
        with st.spinner('Global veriler yükleniyor...'):
            ticker = format_bist(symbol_raw)
            comp_ticker = format_bist(compare_raw)
            
            # Global Veri Kümesi (Geniş aralık)
            df_all = yf.download(ticker, start=g_start, end=g_end, interval="1d")
            df_comp_all = yf.download(comp_ticker, start=g_start, end=g_end, interval="1d")

            if df_all.empty:
                st.error("Veri bulunamadı.")
            else:
                # MultiIndex Fix
                if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
                if isinstance(df_comp_all.columns, pd.MultiIndex): df_comp_all.columns = df_comp_all.columns.get_level_values(0)

                # Temel Hesaplamalar
                df_all['Daily Range'] = (df_all['High'] - df_all['Low']).round(2)
                df_all['Pct_Change'] = df_all['Close'].pct_change() * 100
                df_all['Amihud'] = (df_all['Pct_Change'].abs() / (df_all['Volume'] / 1000000)).round(4)

                # --- 1. ANA GRAFİK (Global Görünüm) ---
                st.subheader(f"📊 {symbol_raw} Global Fiyat Hareketi")
                fig_main = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
                fig_main.add_trace(go.Candlestick(x=df_all.index, open=df_all['Open'], high=df_all['High'], low=df_all['Low'], close=df_all['Close'], name="Fiyat"), row=1, col=1)
                
                # VRP (Global)
                bins = 20
                df_all['PriceBin'] = pd.cut(df_all['Close'], bins=bins)
                v_prof = df_all.groupby('PriceBin', observed=True)['Volume'].sum()
                fig_main.add_trace(go.Bar(x=v_prof.values, y=[i.mid for i in v_prof.index], orientation='h', marker_color='rgba(255, 75, 75, 0.2)', name="Hacim"), row=1, col=2)
                fig_main.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark", showlegend=False, dragmode='pan')
                st.plotly_chart(fig_main, use_container_width=True, config={'scrollZoom': True})

                # --- 2. TARİH DÖNEMİ SEÇME AYRACI (Slider) ---
                st.markdown("---")
                st.subheader("🕒 Zaman Tüneli: Analiz Dönemini Belirle")
                st.info("Aşağıdaki sürgüyü kaydırarak detay grafiklerin ve tablonun hangi tarih aralığını analiz edeceğini seçin.")
                
                # Tarihleri liste olarak alıyoruz (Slider için)
                date_list = df_all.index.tolist()
                
                # Kullanıcının alt grafikler için tarih seçtiği Slider
                start_select, end_select = st.select_slider(
                    'Analiz edilecek tarih aralığını daraltın:',
                    options=date_list,
                    value=(date_list[0], date_list[-1]),
                    format_func=lambda x: x.strftime('%d %b %Y')
                )

                # Seçilen aralığa göre veriyi FİLTRELE
                df = df_all.loc[start_select:end_select].copy()
                df_comp = df_comp_all.loc[start_select:end_select].copy()

                # --- 3. DİNAMİK ANALİZ METRİKLERİ ---
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Seçilen Gün", f"{len(df)}")
                m2.metric("Ort. Daily Range", f"{df['Daily Range'].mean():.2f}")
                m3.metric("Ort. Amihud", f"{df['Amihud'].mean():.4f}")
                corr_val = df['Close'].corr(df_comp['Close'])
                m4.metric("Korelasyon", f"{corr_val:.2f}")

                # --- 4. DETAY VERİ LİSTESİ ---
                st.subheader("📅 Seçilen Dönem Detay Listesi")
                df['Değişim %'] = df['Pct_Change'].apply(lambda x: f"🟢 +%{x:.2f}" if x > 0 else f"🔴 -%{abs(x):.2f}" if x < 0 else "⚪ 0.00")
                st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Range', 'Amihud', 'Değişim %']].sort_index(ascending=False), use_container_width=True, height=350)

                # --- 5. DUAL AXIS GRAFİKLER (Filtrelenmiş Veriyle) ---
                st.divider()
                
                def dual_plot(title, y1, name1, col1, y2, name2, col2, dash=None):
                    f = go.Figure()
                    f.add_trace(go.Scatter(x=df.index, y=y1, name=name1, line=dict(color=col1, width=2.5), yaxis="y1"))
                    f.add_trace(go.Scatter(x=df.index, y=y2, name=name2, line=dict(color=col2, width=2.5, dash=dash), yaxis="y2"))
                    f.update_layout(title=title, template="plotly_dark", height=400,
                                    yaxis=dict(title=name1, tickfont=dict(color=col1)),
                                    yaxis2=dict(title=name2, tickfont=dict(color=col2), anchor="x", overlaying="y", side="right"),
                                    hovermode="x unified", dragmode='pan')
                    st.plotly_chart(f, use_container_width=True, config={'scrollZoom': True})

                dual_plot("Amihud vs Daily Range", df['Amihud'], "Amihud", "#00FFCC", df['Daily Range'], "Daily Range", "#FFD700", dash='dot')
                dual_plot("Daily Range vs Close", df['Close'], "Fiyat", "#FFFFFF", df['Daily Range'], "Daily Range", "#FFD700")
                dual_plot("Amihud vs Close", df['Close'], "Fiyat", "#FFFFFF", df['Amihud'], "Amihud", "#00FFCC")

else:
    st.info("👈 Analizi başlatmak için soldan tarih ve sembol seçip butona basın.")
