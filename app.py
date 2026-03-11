import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from engine import get_portfolio_data, calculate_efficient_frontier, get_tangency_portfolio, get_hrp_weights

st.set_page_config(page_title="Portfolio Analyser", layout="wide")

# --- Sidebar ---
st.sidebar.header("Parameters")
symbols_str = st.sidebar.text_input("Symbols (comma-sep)", "AAPL,MSFT,GOOG,TSLA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-12-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today().normalize())
rf_rate = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.005, format="%.3f")
run_btn = st.sidebar.button("▶ Run Analysis", type="primary")

if run_btn:
    symbols = [s.strip().upper() for s in symbols_str.split(",")]
    try:
        prices, returns = get_portfolio_data(symbols, start_date, end_date)
    except Exception as e:
        st.error(f"資料下載/整理失敗：{e}")
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Risk & Corr", "Efficient Frontier", "Tangency", "HRP"])

    with tab1:
        st.subheader("Monthly Returns & Correlation")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(returns, title="Monthly Returns"), use_container_width=True)
        with col2:
            st.plotly_chart(px.imshow(returns.corr(), text_auto=True, title="Correlation Matrix"), use_container_width=True)
        st.dataframe(returns.describe().T)

    with tab2:
        st.subheader("Efficient Frontier")
        ef_df = calculate_efficient_frontier(returns)
        fig = px.line(ef_df, x="Volatility", y="Return", title="Efficient Frontier (Long-only)")
        # 加入個別資產點
        asset_mu = returns.mean() * 12
        asset_vol = returns.std() * (12**0.5)
        fig.add_trace(go.Scatter(x=asset_vol, y=asset_mu, mode='markers+text', 
                                 text=asset_mu.index, name="Assets"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        tan = get_tangency_portfolio(returns, rf_rate)
        c1, c2, c3 = st.columns(3)
        c1.metric("Ann. Return", f"{tan['ret']:.2%}")
        c2.metric("Ann. Volatility", f"{tan['vol']:.2%}")
        c3.metric("Sharpe Ratio", f"{(tan['ret']-rf_rate)/tan['vol']:.2f}")
        
        w_df = pd.DataFrame({"Asset": symbols, "Weight": tan['weights']})
        st.plotly_chart(px.pie(w_df, values='Weight', names='Asset', title="Tangency Weights"))

    with tab4:
        hrp_w = get_hrp_weights(returns)
        hrp_df = pd.DataFrame(list(hrp_w.items()), columns=['Asset', 'Weight'])
        st.plotly_chart(px.bar(hrp_df, x='Asset', y='Weight', title="HRP Portfolio Weights"))
        
        # Cumulative Returns
        hrp_ret = (returns * pd.Series(hrp_w)).sum(axis=1)
        cum_ret = (1 + hrp_ret).cumprod()
        st.plotly_chart(px.line(cum_ret, title="HRP Cumulative Performance"))