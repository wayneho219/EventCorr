import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pypfopt import risk_models, expected_returns
from pypfopt.hierarchical_portfolio import HRPOpt

def get_portfolio_data(symbols, start_date, end_date):
    """抓取月收盤價並計算報酬率"""
    raw = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    if raw is None or raw.empty:
        raise ValueError(
            "yfinance 沒有下載到任何資料（可能是網路/Proxy/被擋 403、代號錯誤、或日期區間無資料）。"
        )

    # yfinance 可能回傳：
    # - MultiIndex: level0=Price(Adj Close/Close/...) level1=Ticker
    # - MultiIndex: level0=Ticker level1=Price（較少見）
    # - Index: 單一 ticker 的欄位（Adj Close/Close/...）
    def _pick_price(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = set(df.columns.get_level_values(0))
            lvl1 = set(df.columns.get_level_values(1))

            if "Adj Close" in lvl0:
                return df["Adj Close"]
            if "Adj Close" in lvl1:
                return df.xs("Adj Close", level=1, axis=1)
            if "Close" in lvl0:
                return df["Close"]
            if "Close" in lvl1:
                return df.xs("Close", level=1, axis=1)

            raise KeyError("找不到 'Adj Close' 或 'Close' 欄位。")

        # 單一 ticker：一般欄位
        if "Adj Close" in df.columns:
            return df[["Adj Close"]].rename(columns={"Adj Close": symbols[0] if symbols else "Adj Close"})
        if "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": symbols[0] if symbols else "Close"})

        raise KeyError("找不到 'Adj Close' 或 'Close' 欄位。")

    data = _pick_price(raw).dropna(how="all")
    returns = data.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("報酬率計算結果為空（可能資料筆數不足或全部缺值）。")
    return data, returns

def calculate_efficient_frontier(returns, n_points=50):
    """計算效率前緣曲線"""
    mu = returns.mean() * 12
    S = returns.cov() * 12
    n_assets = len(mu)
    
    def portfolio_stats(weights):
        p_ret = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        return p_ret, p_vol

    target_rets = np.linspace(mu.min(), mu.max(), n_points)
    frontier_vols = []
    
    for t in target_rets:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - t})
        bounds = tuple((0, 1) for _ in range(n_assets))
        res = minimize(lambda x: portfolio_stats(x)[1], n_assets*[1./n_assets], 
                       method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            frontier_vols.append(res.fun)
        else:
            frontier_vols.append(None)
            
    return pd.DataFrame({'Return': target_rets, 'Volatility': frontier_vols}).dropna()

def get_tangency_portfolio(returns, rf=0.02):
    """計算切點投資組合 (Max Sharpe Ratio)"""
    mu = returns.mean() * 12
    S = returns.cov() * 12
    n_assets = len(mu)
    
    def neg_sharpe(weights):
        p_ret = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        return -(p_ret - rf) / p_vol

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    res = minimize(neg_sharpe, n_assets*[1./n_assets], method='SLSQP', bounds=bounds, constraints=cons)
    
    return {"weights": res.x, "ret": np.sum(mu * res.x), "vol": np.sqrt(np.dot(res.x.T, np.dot(S, res.x)))}

def get_hrp_weights(returns):
    """計算階層風險預算 (HRP)"""
    hrp = HRPOpt(returns)
    return hrp.optimize()