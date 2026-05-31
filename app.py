import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# إعدادات صفحة الويب
st.set_page_config(page_title="منصة الذكاء الاصطناعي الأخلاقي (IEAI)", layout="wide", initial_sidebar_state="expanded")

# توجيه النص ليكون من اليمين لليسار (RTL) لدعم العربية بشكل مثالي
st.markdown("""
<style>
    body, p, h1, h2, h3, h4, h5, h6, .stMarkdown {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌙 منصة الذكاء الاصطناعي الأخلاقي الإسلامي (IEAI)")
st.subheader("محاكاة تفاعلية - (AEWS) نظام الأوزان الأخلاقية المتكيفة")
st.markdown("---")

# ==========================================
# 1. الشريط الجانبي (إعدادات المستخدم)
# ==========================================
st.sidebar.header("⚙️ إعدادات المنصة")
start_date = st.sidebar.date_input("تاريخ البداية", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية", datetime.date(2026, 1, 1))
window_size = st.sidebar.slider("نافذة القياس (أيام)", min_value=21, max_value=126, value=63)

# ==========================================
# 2. تحميل البيانات (مع التخزين المؤقت لتسريع المنصة)
# ==========================================
@st.cache_data
def load_data(start, end):
    tickers = ['THYAO.IS', 'AKBNK.IS', 'TUPRS.IS', 'BIMAS.IS', 'ASELS.IS', 'HEKTS.IS']
    market = 'XU100.IS'
    all_symbols = tickers + [market]
    data = yf.download(all_symbols, start=start, end=end, auto_adjust=False, progress=False)
    
    adj_close = data['Adj Close'].ffill()
    high = data['High'].ffill()
    low = data['Low'].ffill()
    volume = data['Volume'].ffill()
    
    return adj_close, high, low, volume, tickers, market

try:
    with st.spinner("جاري جلب البيانات اللحظية وإجراء الحسابات المعقدة..."):
        adj_close, high, low, volume, tickers, market = load_data(start_date, end_date)
        
        market_returns = adj_close[market].pct_change().fillna(0)
        stock_returns = adj_close[tickers].pct_change().fillna(0)

        # ==========================================
        # 3. محرك الحسابات المقاصدية و AEWS
        # ==========================================
        vol_W = stock_returns.rolling(window=window_size).std() * np.sqrt(252)
        rolling_max = adj_close[tickers].rolling(window=window_size, min_periods=1).max()
        mdd_W = ((adj_close[tickers] - rolling_max) / rolling_max).rolling(window=window_size, min_periods=1).min().abs()
        spread_proxy = (high[tickers] - low[tickers]) / adj_close[tickers]
        spread_M = spread_proxy.rolling(window=window_size).mean()
        amihud = stock_returns.abs() / volume[tickers]
        amihud_J = amihud.rolling(window=window_size).mean() * 1e6
        
        cov = stock_returns.rolling(window_size).cov(market_returns)
        var = market_returns.rolling(window_size).var()
        beta_N = cov.div(var, axis=0)

        def normalize(df):
            df_min = df.min(axis=1)
            df_max = df.max(axis=1)
            range_diff = (df_max - df_min).replace(0, np.nan)
            return df.subtract(df_max, axis=0).abs().div(range_diff, axis=0) * 100

        score_W = (normalize(vol_W) * 0.5) + (normalize(mdd_W) * 0.5)
        score_M = normalize(spread_M)
        score_J = normalize(amihud_J)
        score_N = normalize((beta_N - 1).abs())
        
        dar_scores = {'THYAO.IS': 40, 'AKBNK.IS': 20, 'TUPRS.IS': 50, 'BIMAS.IS': 90, 'ASELS.IS': 85, 'HEKTS.IS': 80}
        score_D = pd.DataFrame(index=score_W.index, columns=tickers)
        for t in tickers: score_D[t] = dar_scores[t]

        market_vol_21 = market_returns.rolling(21).std() * np.sqrt(252)
        m_vol_mean_252 = market_vol_21.rolling(252).mean()
        m_vol_std_252 = market_vol_21.rolling(252).std()
        z_score = (market_vol_21 - m_vol_mean_252) / m_vol_std_252

        ieacs_stable = score_W*0.20 + score_M*0.20 + score_J*0.20 + score_N*0.10 + score_D*0.30
        ieacs_volatile = score_W*0.15 + score_M*0.35 + score_J*0.25 + score_N*0.10 + score_D*0.15
        ieacs_crisis = score_W*0.45 + score_M*0.05 + score_J*0.05 + score_N*0.35 + score_D*0.10

        ieacs_dynamic = ieacs_stable.copy()
        for col in tickers:
            ieacs_dynamic[col] = np.where(z_score <= 1, ieacs_stable[col],
                                 np.where((z_score > 1) & (z_score <= 2.5), ieacs_volatile[col],
                                 np.where(z_score > 2.5, ieacs_crisis[col], np.nan)))

        ieacs_shifted = ieacs_dynamic.shift(1)
        top_2_mask = ieacs_shifted.rank(axis=1, ascending=False, method='first') <= 2
        portfolio_returns = stock_returns[top_2_mask].mean(axis=1).fillna(0)

        # ==========================================
        # 4. لوحة المؤشرات العلوية (Dashboard Metrics)
        # ==========================================
        st.markdown("### 📊 (AEWS) رادار حالة السوق")
        current_z = z_score.iloc[-1]
        if pd.isna(current_z):
            market_state = "جاري الحساب..."
            color = "gray"
        elif current_z <= 1:
            market_state = "استقرار (Stable) 🟢"
            color = "normal"
        elif current_z <= 2.5:
            market_state = "تقلب (Volatile) 🟡"
            color = "off"
        else:
            market_state = "أزمة (Crisis) 🔴"
            color = "inverse"

        col1, col2, col3 = st.columns(3)
        col1.metric("حالة السوق الحالية", market_state)
        col2.metric("مؤشر التقلب (Z-Score)", f"{current_z:.2f}" if not pd.isna(current_z) else "N/A")
        col3.metric("عدد الأسهم تحت المراقبة", str(len(tickers)))

        st.markdown("---")

        # ==========================================
        # 5. الرسم البياني التفاعلي (Plotly)
        # ==========================================
        st.markdown("### 📈 الأداء التراكمي: المحفظة الأخلاقية مقابل السوق")
        
        start_viz = str(start_date)
        port_cum = (1 + portfolio_returns.loc[start_viz:]).cumprod()
        mkt_cum = (1 + market_returns.loc[start_viz:]).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, mode='lines', name='(IEACS) المحفظة الأخلاقية', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=mkt_cum.index, y=mkt_cum, mode='lines', name='(BIST 100) المؤشر العام', line=dict(color='red', width=1, dash='dot')))
        
        fig.update_layout(
            title="(Backtest) مقارنة النمو التراكمي",
            xaxis_title="التاريخ",
            yaxis_title="مضاعف رأس المال",
            template="plotly_dark", # ثيم احترافي
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("يرجى اختيار نطاق زمني أوسع للسماح للنموذج بحساب النافذة الزمنية بدقة.")
