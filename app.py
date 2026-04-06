import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Ayarları
st.set_page_config(page_title="BIST Global Terminal", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; height: 3.5em; }
    input { text-transform: uppercase; }
    .main { overflow: auto; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ BIST Tarihsel Analiz & Likidite Terminali")

# --- SIDEBAR (PARAMETRE KONSOLU) ---
with st.sidebar:
    st.header("🔍 Analiz Parametreleri")
    symbol_raw = st.text_input("Sembol", placeholder="Örn: THYAO").strip().upper()
    compare_raw = st.text_input("Korelasyon Sembolü", value="XU100").strip().upper()
    
    st.divider()
    st.subheader("📅 Geniş Veri Aralığı")
    
    # Bugün 2026, başlangıcı 1990'a kadar çekiyoruz (Yahoo'nun sınırı)
    min_date = datetime(1990, 1, 1)
    max_date = datetime.now()
    
    # Varsayılan başlangıç: 1 yıl öncesi olsun ama kullanıcı 1990'a kadar inebilir
    default_start = datetime.now() - timedelta(days=365)
    
    start_date = st.date_input("Başlangıç Tarihi", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.date_input("Bitiş Tarihi", value=max_date, min_value=min_date, max_value=max_date)
    
    if start_date > end_date:
        st.error("Hata: Başlangıç tarihi bitişten büyük olamaz.")
        
    st.divider()
    run_analysis = st.button("ANALİZİ BAŞLAT")

def format_bist(s):
    if not s: return None
    return f"{s}.IS" if not s.endswith(".IS") else s

if run_analysis:
    if not symbol_raw:
        st.warning("⚠️ Lütfen bir sembol giriniz.")
    else:
        with st.spinner(f'{symbol_raw} tarihsel verileri işleniyor...'):
            ticker = format_bist(symbol_raw)
            comp_ticker = format_bist(compare_raw)
            
            # Veriyi seçilen tüm aralıkta çekiyoruz
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            df_comp = yf.download(comp_ticker, start=start_date, end=end_date, interval="1d")

            if df.empty or len(df) < 5:
                st.error(f"❌ {symbol_raw} için bu aralıkta yeterli veri bulunamadı. (Yahoo'da veri başlamamış olabilir)")
            else:
                # MultiIndex Fix
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                if isinstance(df_comp.columns, pd.MultiIndex): df_comp.columns = df_comp.columns.get_level_values(0)

                # --- HESAPLAMALAR ---
                df['Daily Range'] = (df['High'] - df['Low']).round(2)
                df['Pct_Change'] = df['Close'].pct_change() * 100
                df['Amihud'] = (df['Pct_Change'].abs() / (df['Volume'] / 1000000)).round(4)
                
                # --- 1. ANA FİYAT GRAFİĞİ VE VRP ---
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    st.subheader(f"📊 {symbol_raw} - Fiyat Hareketi")
                    fig_main = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
                    fig_main.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
                    
                    # Tüm seçilen aralık için Hacim Profili
                    bins = 20
                    df_clean = df.dropna(subset=['Close', 'Volume'])
                    df_clean['PriceBin'] = pd.cut(df_clean['Close'], bins=bins)
                    vprofile = df_clean.groupby('PriceBin', observed=True)['Volume'].sum()
                    fig_main.add_trace(go.Bar(x=vprofile.values, y=[i.mid for i in vprofile.index], orientation='h', marker_color='rgba(255, 75, 75, 0.3)', name="Hacim"), row=1, col=2)
                    
                    # Başlangıç görünümü son 100 gün olsun ama kullanıcı geri kaydırabilsin
                    start_view = df.index[-100] if len(df) > 100 else df.index[0]
                    fig_main.update_layout(xaxis_rangeslider_visible=False, height=550, template="plotly_dark", showlegend=False, dragmode='pan', xaxis=dict(range=[start_view, df.index[-1]]))
                    st.plotly_chart(fig_main, use_container_width=True, config={'scrollZoom': True})

                with col_right:
                    st.subheader("🔗 İstatistikler")
                    combined = pd.concat([df['Close'], df_comp['Close']], axis=1).dropna()
                    combined.columns = ['Hisse', 'Endeks']
                    if not combined.empty:
                        corr = combined['Hisse'].corr(combined['Endeks'])
                        st.metric(f"Korelasyon ({compare_raw})", f"{corr:.2f}")
                    
                    st.write(f"**Aralık Ort. Amihud:** {df['Amihud'].mean():.4f}")
                    st.write(f"**Aralık Ort. Range:** {df['Daily Range'].mean():.2f}")
                    st.write(f"**Toplam İşlem Günü:** {len(df)}")

                # --- 2. TABLO ---
                st.divider()
                st.subheader("📅 Detay Veri Listesi (Tarihsel)")
                res_df = df.copy()
                res_df['Değişim %'] = res_df['Pct_Change'].apply(lambda x: f"🟢 +%{x:.2f}" if x > 0 else f"🔴 -%{abs(x):.2f}" if x < 0 else "⚪ 0.00")
                # Tablo sadece son 100 günü göstersin (Performans için), istersen hepsini gösterebilirsin
                st.dataframe(res_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Range', 'Amihud', 'Değişim %']].sort_index(ascending=False), use_container_width=True, height=400)

                # --- 3. DUAL AXIS ANALİZ GRAFİKLERİ ---
                st.divider()
                def create_dual_chart(title, y1_data, y1_name, y1_color, y2_data, y2_name, y2_color, dash=None):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=y1_data, name=y1_name, line=dict(color=y1_color, width=2.5), yaxis="y1"))
                    fig.add_trace(go.Scatter(x=df.index, y=y2_data, name=y2_name, line=dict(color=y2_color, width=2.5, dash=dash), yaxis="y2"))
                    fig.update_layout(title=title, template="plotly_dark", height=450, 
                                     yaxis=dict(title=dict(text=y1_name, font=dict(color=y1_color)), tickfont=dict(color=y1_color)),
                                     yaxis2=dict(title=dict(text=y2_name, font=dict(color=y2_color)), tickfont=dict(color=y2_color), anchor="x", overlaying="y", side="right"),
                                     hovermode="x unified", dragmode='pan')
                    return fig

                st.plotly_chart(create_dual_chart("Amihud vs Daily Range", df['Amihud'], "Amihud", "#00FFCC", df['Daily Range'], "Daily Range", "#FFD700", dash='dot'), use_container_width=True, config={'scrollZoom': True})
                st.plotly_chart(create_dual_chart("Daily Range vs Close", df['Close'], "Fiyat (Close)", "#FFFFFF", df['Daily Range'], "Daily Range", "#FFD700"), use_container_width=True, config={'scrollZoom': True})
                st.plotly_chart(create_dual_chart("Amihud vs Close", df['Close'], "Fiyat (Close)", "#FFFFFF", df['Amihud'], "Amihud (Likidite)", "#00FFCC"), use_container_width=True, config={'scrollZoom': True})

else:
    st.info("👈 Takvimi 1990'a kadar çekebilirsiniz. Analiz için sembol girin ve butona basın.")
