#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCA 投資模擬器 — tkinter 互動版
功能：自行輸入股票代號與參數 · IRR · CAGR · Max Drawdown · 自動匯出 HTML 報告
"""

# ── 自動安裝依賴套件 ──────────────────────────────────────
import subprocess, sys

for pkg, imp in [('numpy_financial','numpy_financial'),('yfinance','yfinance'),
                 ('matplotlib','matplotlib')]:
    try: __import__(imp)
    except ImportError: subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])

# ── 匯入 ─────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox
import threading, json, os, webbrowser

import numpy as np
import numpy_financial as npf  # type: ignore  (動態安裝後可用)
import yfinance as yf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── 修正中文字體亂碼 ──────────────────────────────────────
for font in ['Microsoft YaHei','SimHei','STHeiti','PingFang SC']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.family'] = font
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

# ── 色彩主題 ──────────────────────────────────────────────
BG    = '#0f1117'
SURF  = '#1a1d27'
SURF2 = '#232636'
BORD  = '#2e3250'
ACC   = '#5b8dee'
GRN   = '#4ade80'
RED   = '#f87171'
TXT   = '#e2e8f0'
MUTED = '#8892a4'

# ═════════════════════════════════════════════════════════
# HTML 報告模板（資料嵌入，瀏覽器無需網路）
# ═════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html><html lang="zh-TW">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCA 報告 – __TICKER__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--sf:#1a1d27;--sf2:#232636;--bd:#2e3250;--ac:#5b8dee;--gn:#4ade80;--rd:#f87171;--tx:#e2e8f0;--mu:#8892a4}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;padding:24px 16px}
h1{font-size:1.5rem;font-weight:700;color:var(--ac);margin-bottom:3px}
.sub{color:var(--mu);font-size:.83rem;margin-bottom:22px}
.con{max-width:1100px;margin:0 auto}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:18px;margin-bottom:18px;box-shadow:0 4px 20px rgba(0,0,0,.4)}
h2{font-size:.92rem;font-weight:600;margin-bottom:12px}
.mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:11px;margin-bottom:18px}
.mc{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.4)}
.mc .l{color:var(--mu);font-size:.7rem;letter-spacing:.05em;margin-bottom:4px}
.mc .v{font-size:1.25rem;font-weight:700}
.mc .s{font-size:.68rem;color:var(--mu);margin-top:3px}
.gn{color:var(--gn)}.rd{color:var(--rd)}.ac{color:var(--ac)}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead th{background:var(--sf2);color:var(--mu);font-weight:600;padding:8px 10px;text-align:right;border-bottom:1px solid var(--bd);white-space:nowrap}
thead th:first-child{text-align:center}
tbody tr:hover{background:rgba(91,141,238,.07)}
tbody td{padding:6px 10px;border-bottom:1px solid rgba(46,50,80,.4);text-align:right}
tbody td:first-child{text-align:center;color:var(--mu);font-weight:600}
.pos{color:var(--gn)}.neg{color:var(--rd)}
.cg{display:grid;grid-template-columns:1fr;gap:18px}
@media(min-width:760px){.cg{grid-template-columns:1fr 1fr}}
.cb{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px}
.ct{font-size:.86rem;font-weight:600;margin-bottom:11px;color:var(--mu)}
canvas{max-height:260px}
footer{text-align:center;color:var(--mu);font-size:.73rem;margin-top:26px}
</style></head>
<body><div class="con">
<h1>📈 DCA 投資模擬器</h1>
<p class="sub">股票代號：<strong style="color:var(--tx)">__TICKER__</strong> · __START__–__END__ · 每年投入 $__INV__</p>
<div id="mg" class="mg"></div>
<div class="card"><h2>年度明細</h2><div class="tw"><table>
<thead><tr><th>年份</th><th>年初投入</th><th>報酬率(%)</th><th>當年損益</th><th>年底市值</th><th>累計投入</th><th>回撤(%)</th></tr></thead>
<tbody id="tb"></tbody></table></div></div>
<div class="cg">
<div class="cb"><div class="ct">組合市值 vs 累計投入</div><canvas id="c1"></canvas></div>
<div class="cb"><div class="ct">年度報酬率(%)</div><canvas id="c2"></canvas></div>
<div class="cb" style="grid-column:1/-1"><div class="ct">回撤走勢(%)</div><canvas id="c3"></canvas></div>
</div>
<footer>資料由 Python/yfinance 產生 · 本工具僅供學術模擬，不構成投資建議</footer>
</div>
<script>
const D=__DATA__;
const fm=(n,d=2)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pt=(n,d=2)=>n.toFixed(d)+'%';
const cl=n=>n>=0?'gn':'rd';
const profit=D.finalWealth-D.totalInvested;
[{l:'最終市值',v:'$'+fm(D.finalWealth),s:'',c:'ac'},
 {l:'總投入',v:'$'+fm(D.totalInvested),s:'',c:''},
 {l:'總損益',v:(profit>=0?'+':'')+'$'+fm(profit),s:'',c:cl(profit)},
 {l:'IRR',v:pt(D.irr),s:'最精確DCA年化',c:cl(D.irr)},
 {l:'CAGR（市場）',v:pt(D.cagrP),s:'市場TWR年化',c:cl(D.cagrP)},
 {l:'CAGR（倍數）',v:pt(D.cagrI),s:'final/invested',c:cl(D.cagrI)},
 {l:'Max Drawdown',v:pt(D.maxDD),s:D.maxDDY+'年',c:'rd'},
].forEach(m=>{
  document.getElementById('mg').insertAdjacentHTML('beforeend',
  `<div class="mc"><div class="l">${m.l}</div><div class="v ${m.c}">${m.v}</div>${m.s?`<div class="s">${m.s}</div>`:''}</div>`);
});
D.years.forEach((yr,i)=>{
  const rc=D.rets[i]>=0?'pos':'neg',gl=D.gains[i]>=0?'pos':'neg',dc=D.dds[i]<-0.01?'neg':'pos';
  document.getElementById('tb').insertAdjacentHTML('beforeend',
  `<tr><td>${yr}</td><td>${fm(D.annInv)}</td><td class="${rc}">${D.rets[i].toFixed(2)}</td>
   <td class="${gl}">${(D.gains[i]>=0?'+':'')+fm(D.gains[i])}</td>
   <td>${fm(D.hist[i])}</td><td>${fm(D.cum[i])}</td>
   <td class="${dc}">${D.dds[i].toFixed(2)}</td></tr>`);
});
const co=(unit,zero=false)=>({responsive:true,animation:{duration:500},
  plugins:{legend:{labels:{color:'#8892a4',boxWidth:12,font:{size:11}}},
    tooltip:{backgroundColor:'#232636',borderColor:'#2e3250',borderWidth:1,
      titleColor:'#e2e8f0',bodyColor:'#94a3b8',
      callbacks:{label:c=>` ${c.dataset.label}: ${c.parsed.y.toLocaleString()}${unit}`}}},
  scales:{x:{ticks:{color:'#8892a4',font:{size:10}},grid:{color:'rgba(255,255,255,.04)'}},
    y:{ticks:{color:'#8892a4',font:{size:10},callback:v=>v.toLocaleString()+unit},
      grid:{color:'rgba(255,255,255,.06)'},
      ...(zero?{afterDataLimits:s=>{s.min=Math.min(s.min,0);}}:{})}}});
new Chart(document.getElementById('c1'),{type:'line',data:{labels:D.years,datasets:[
  {label:'組合市值',data:D.hist,borderColor:'#5b8dee',backgroundColor:'rgba(91,141,238,.15)',fill:true,tension:.3,pointRadius:3,borderWidth:2},
  {label:'累計投入',data:D.cum,borderColor:'#64748b',borderDash:[5,4],fill:false,tension:0,pointRadius:0,borderWidth:1.5}
]},options:co('$')});
new Chart(document.getElementById('c2'),{type:'bar',data:{labels:D.years,datasets:[
  {label:'報酬率',data:D.rets,borderRadius:3,backgroundColor:D.rets.map(r=>r>=0?'rgba(74,222,128,.7)':'rgba(248,113,113,.7)')}
]},options:co('%',true)});
new Chart(document.getElementById('c3'),{type:'bar',data:{labels:D.years,datasets:[
  {label:'回撤',data:D.dds,borderRadius:3,backgroundColor:D.dds.map(d=>d<-0.01?'rgba(248,113,113,.75)':'rgba(74,222,128,.45)')}
]},options:co('%',true)});
</script></body></html>"""


# ═════════════════════════════════════════════════════════
# 主應用程式類別
# ═════════════════════════════════════════════════════════
class DCAApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('📈  DCA 投資模擬器')
        self.root.configure(bg=BG)
        self.root.geometry('1300x860')
        self.root.minsize(960, 640)
        self._build_styles()
        self._build_ui()

    # ── 樣式設定 ─────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self.root)
        s.theme_use('clam')
        s.configure('.',             background=BG,    foreground=TXT, font=('Segoe UI', 10))
        s.configure('TFrame',        background=BG)
        s.configure('TLabel',        background=BG,    foreground=TXT)
        s.configure('TScrollbar',    background=SURF2, troughcolor=BG, bordercolor=BORD, arrowcolor=MUTED)
        s.configure('Treeview',      background=SURF2, foreground=TXT, fieldbackground=SURF2,
                     rowheight=22,   font=('Segoe UI', 9))
        s.configure('Treeview.Heading', background=SURF, foreground=MUTED,
                     font=('Segoe UI', 9, 'bold'), relief='flat')
        s.map('Treeview', background=[('selected', ACC)], foreground=[('selected', 'white')])

    # ── UI 佈局 ───────────────────────────────────────────
    def _build_ui(self):
        # ── 頂部標題 ──────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill='x', padx=20, pady=(16, 4))
        tk.Label(hdr, text='📈  DCA 投資模擬器',
                 bg=BG, fg=ACC, font=('Segoe UI', 17, 'bold')).pack(side='left')
        tk.Label(hdr, text='定期定額 · IRR · CAGR · Max Drawdown',
                 bg=BG, fg=MUTED, font=('Segoe UI', 10)).pack(side='left', padx=14, pady=(5, 0))

        # ── 參數輸入卡 ────────────────────────────────────
        pc = tk.Frame(self.root, bg=SURF, pady=12, padx=16)
        pc.pack(fill='x', padx=20, pady=(0, 6))

        self.v_ticker = self._param_entry(pc, '股票代號 (Ticker)', '2330.TW', 10)
        self.v_start  = self._param_entry(pc, '起始年份',          '2003',     7)
        self.v_end    = self._param_entry(pc, '結束年份',          '2024',     7)
        self.v_inv    = self._param_entry(pc, '每期投入金額 ($)',   '10000',   10)

        # 週期選擇
        pf = tk.Frame(pc, bg=SURF)
        pf.pack(side='left', padx=(0, 14))
        tk.Label(pf, text='計算週期', bg=SURF, fg=MUTED,
                 font=('Segoe UI', 8)).pack(anchor='w')
        self.v_period = tk.StringVar(value='Y')
        rbf = tk.Frame(pf, bg=SURF)
        rbf.pack(anchor='w')
        for txt, val in [('按年', 'Y'), ('按月', 'M')]:
            tk.Radiobutton(rbf, text=txt, variable=self.v_period, value=val,
                           bg=SURF, fg=TXT, selectcolor=SURF2,
                           activebackground=SURF, activeforeground=ACC,
                           font=('Segoe UI', 10)).pack(side='left', padx=(0, 8))

        # 按鈕欄
        bf = tk.Frame(pc, bg=SURF)
        bf.pack(side='left', padx=(6, 0))
        self._btn(bf, '▶  執行模擬', self._on_run,  ACC,  '#4a7adc').pack(pady=(16, 2))
        self._btn(bf, '🌐  匯出報告', self._on_export, SURF2, BORD, border=True).pack()

        # ── 狀態列 ────────────────────────────────────────
        self.status_var = tk.StringVar(value='請輸入參數後按「執行模擬」')
        self._status_lbl = tk.Label(self.root, textvariable=self.status_var,
                                    bg=SURF2, fg=MUTED, font=('Segoe UI', 9),
                                    anchor='w', padx=12, pady=5)
        self._status_lbl.pack(fill='x', padx=20)

        # ── 指標卡片列 ────────────────────────────────────
        mf = tk.Frame(self.root, bg=BG)
        mf.pack(fill='x', padx=20, pady=6)
        self._metric_val = {}
        for key in ['最終市值','總投入','總損益','IRR','CAGR(市場)','CAGR(倍數)','Max DD']:
            card = tk.Frame(mf, bg=SURF, padx=10, pady=8)
            card.pack(side='left', expand=True, fill='both', padx=(0, 5))
            tk.Label(card, text=key, bg=SURF, fg=MUTED,
                     font=('Segoe UI', 8)).pack()
            lbl = tk.Label(card, text='—', bg=SURF, fg=MUTED,
                           font=('Segoe UI', 13, 'bold'))
            lbl.pack()
            self._metric_val[key] = lbl

        # ── 主體：圖表 ＋ 表格 ────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=20, pady=(0, 10))

        # 圖表區
        cf = tk.Frame(body, bg=SURF)
        cf.pack(side='left', fill='both', expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(9, 5.5),
            gridspec_kw={'height_ratios': [3, 1]},
            facecolor=SURF
        )
        self.fig.subplots_adjust(hspace=0.35, left=0.11, right=0.97, top=0.91, bottom=0.09)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        tb_frame = tk.Frame(cf, bg=SURF)
        tb_frame.pack(fill='x')
        NavigationToolbar2Tk(self.canvas, tb_frame)

        # 表格區
        tf = tk.Frame(body, bg=SURF, width=360)
        tf.pack(side='right', fill='y', padx=(7, 0))
        tf.pack_propagate(False)

        tk.Label(tf, text='年度明細', bg=SURF, fg=TXT,
                 font=('Segoe UI', 10, 'bold'), padx=10, pady=8).pack(anchor='w')

        cols = ('年份','報酬率%','損益','年底市值','回撤%')
        self.tree = ttk.Treeview(tf, columns=cols, show='headings')
        widths = [48, 68, 82, 88, 62]
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='e' if c != '年份' else 'center', stretch=True)
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True, padx=(8, 0), pady=(0, 8))
        sb.pack(side='right', fill='y', pady=(0, 8))

        self._sim_result = None   # 保留最後模擬結果供匯出

    # ── 小工具 ───────────────────────────────────────────
    def _param_entry(self, parent, label, default, width):
        f = tk.Frame(parent, bg=SURF)
        f.pack(side='left', padx=(0, 16))
        tk.Label(f, text=label, bg=SURF, fg=MUTED,
                 font=('Segoe UI', 8)).pack(anchor='w')
        var = tk.StringVar(value=default)
        tk.Entry(f, textvariable=var, width=width,
                 bg=SURF2, fg=TXT, insertbackground=TXT,
                 relief='flat', bd=5, font=('Segoe UI', 11),
                 highlightthickness=1, highlightcolor=ACC,
                 highlightbackground=BORD).pack()
        return var

    def _btn(self, parent, text, cmd, bg, active_bg, border=False):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=TXT if border else 'white',
                         activebackground=active_bg,
                         font=('Segoe UI', 10, 'bold'),
                         relief='flat', bd=0, padx=14, pady=5,
                         cursor='hand2')

    def _style_axes(self):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor(SURF)
            ax.tick_params(colors=MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(BORD)
            ax.grid(True, color=BORD, alpha=0.45, linewidth=0.55)
        self.ax1.set_title('請輸入參數後按「執行模擬」', color=MUTED, fontsize=10)
        if hasattr(self, 'canvas'):
            self.canvas.draw()

    def _set_status(self, msg, color=MUTED):
        self._status_lbl.configure(fg=color)
        self.status_var.set(msg)

    # ── 執行模擬 ─────────────────────────────────────────
    def _on_run(self):
        threading.Thread(target=self._run_simulation, daemon=True).start()

    def _run_simulation(self):
        ticker      = self.v_ticker.get().strip().upper()
        period_mode = self.v_period.get()          # 'Y' = 年, 'M' = 月
        try:
            start_y  = int(self.v_start.get())
            end_y    = int(self.v_end.get())
            inv_amt  = float(self.v_inv.get())
            if start_y >= end_y: raise ValueError
        except ValueError:
            self._set_status('❌ 參數格式錯誤（年份需為整數且起始 < 結束）', RED)
            return

        unit     = '月' if period_mode == 'M' else '年'
        freq     = 'ME' if period_mode == 'M' else 'YE'
        ppy      = 12   if period_mode == 'M' else 1    # periods per year

        self._set_status(f'⏳ 正在下載 {ticker} 資料（按{unit}）...', ACC)

        try:
            raw = yf.download(ticker,
                              start=f'{start_y - 1}-01-01',
                              end=f'{end_y}-12-31',
                              interval='1d', progress=False)
            if raw.empty:
                raise ValueError('找不到資料，請確認股票代號是否正確')

            prices = raw['Close'].resample(freq).last()
            rets   = prices.pct_change().dropna()

            # 過濾年份範圍
            mask   = [(start_y <= ts.year <= end_y) for ts in rets.index]
            rets   = rets[mask]
            if rets.empty:
                raise ValueError('指定期間內無資料')

            # 週期標籤
            if period_mode == 'M':
                labels = [ts.strftime('%Y-%m') for ts in rets.index]
            else:
                labels = [ts.year for ts in rets.index]

            spy_r = rets.to_numpy().flatten()

        except Exception as e:
            self._set_status(f'❌ 資料下載失敗：{e}', RED)
            return

        self._set_status('⚙️ 計算中...', ACC)

        # ── 財富累積模擬 ──────────────────────────────────
        portfolio, cash_flows, rows = 0.0, [], []
        peak = 0.0
        for i, (lbl, ret) in enumerate(zip(labels, spy_r)):
            portfolio += inv_amt
            cash_flows.append(-inv_amt)
            after_inv  = portfolio
            portfolio *= (1 + float(ret))
            gain       = portfolio - after_inv
            if portfolio > peak: peak = portfolio
            dd = (portfolio - peak) / peak if peak > 0 else 0.0
            rows.append(dict(label=lbl, ret=float(ret), gain=gain,
                             val=portfolio, dd=dd, cum=inv_amt*(i+1)))

        final_wealth   = portfolio
        total_invested = inv_amt * len(rows)
        cash_flows.append(final_wealth)

        # ── 指標計算（月度 IRR/CAGR 均年化）────────────────
        history  = [r['val'] for r in rows]
        n        = len(rows)

        # IRR：考慮每期現金流時機，最精確的 DCA 投資人報酬
        irr_raw  = float(npf.irr(np.array(cash_flows)))
        irr      = (1 + irr_raw) ** ppy - 1                          # 年化 IRR

        # CAGR(市場)：時間加權報酬率 TWR，連乘每期報酬，排除追加資金干擾
        # 反映「若持有 $1 不動的市場年化表現」，與 IRR 對比可看出 DCA 效益
        twr = 100.0
        for ret in spy_r:
            twr *= (1 + float(ret))
        cagr_mkt = twr ** (ppy / n) - 1                              # 年化市場 TWR

        # CAGR(倍數)：(最終市值/累計投入)^(1/年數)，粗略衡量資金增長速度
        # 注意：分母是「分批」投入的，並非一次投入，故偏低於 IRR
        cagr_i   = (final_wealth/total_invested) ** (ppy/n) - 1      # 年化 CAGR(倍數)

        dds_arr  = np.array([r['dd'] for r in rows])
        max_dd   = float(dds_arr.min())
        max_dd_lbl = rows[int(np.argmin(dds_arr))]['label']

        # X 軸刻度：月資料只顯示每年一個刻度
        if period_mode == 'M':
            x_vals   = list(range(n))
            tick_idx = [i for i in range(0, n, 12)]
            x_ticks  = tick_idx
            x_tlbls  = [str(labels[i])[:4] for i in tick_idx]
            x_lbl    = '月份'
        else:
            x_vals  = labels
            x_ticks = labels
            x_tlbls = [str(y) for y in labels]
            x_lbl   = '年份'

        self._sim_result = dict(
            rows=rows, labels=labels, history=history,
            irr=irr, cagr=cagr_mkt, cagr_i=cagr_i,
            max_dd=max_dd, max_dd_lbl=max_dd_lbl,
            final_wealth=final_wealth, total_invested=total_invested,
            inv_amt=inv_amt, ticker=ticker, unit=unit, ppy=ppy,
            x_vals=x_vals, x_ticks=x_ticks, x_tlbls=x_tlbls, x_lbl=x_lbl,
        )

        self.root.after(0, self._update_ui, self._sim_result)

    # ── UI 更新（必須在主執行緒）──────────────────────────
    def _update_ui(self, r):
        rows         = r['rows']
        labels       = r['labels']
        history      = r['history']
        irr          = r['irr']
        cagr         = r['cagr']
        cagr_i       = r['cagr_i']
        max_dd       = r['max_dd']
        max_dd_lbl   = r['max_dd_lbl']
        final_wealth = r['final_wealth']
        total_inv    = r['total_invested']
        inv_amt      = r['inv_amt']
        ticker       = r['ticker']
        unit         = r['unit']
        x_vals       = r['x_vals']
        x_ticks      = r['x_ticks']
        x_tlbls      = r['x_tlbls']
        x_lbl        = r['x_lbl']
        profit       = final_wealth - total_inv
        n            = len(rows)

        # ── 指標卡片 ──────────────────────────────────────
        def fmt(v): return f'${v:,.0f}'
        def pct(v): return f'{v*100:.2f}%'
        updates = {
            '最終市值':   (fmt(final_wealth),                      ACC),
            '總投入':     (fmt(total_inv),                         TXT),
            '總損益':     (('+' if profit>=0 else '')+fmt(profit), GRN if profit>=0 else RED),
            'IRR':        (pct(irr),                               GRN if irr>=0 else RED),
            'CAGR(市場)': (pct(cagr),                              GRN if cagr>=0 else RED),
            'CAGR(倍數)': (pct(cagr_i),                            GRN if cagr_i>=0 else RED),
            'Max DD':     (pct(max_dd),                            RED),
        }
        for key, (val, col) in updates.items():
            self._metric_val[key].configure(text=val, fg=col)

        # ── 明細表格 ──────────────────────────────────────
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in rows:
            tag = 'pos' if row['ret'] >= 0 else 'neg'
            self.tree.insert('', 'end', values=(
                row['label'],
                f"{row['ret']*100:.2f}",
                f"{row['gain']:+,.0f}",
                f"{row['val']:,.0f}",
                f"{row['dd']*100:.2f}",
            ), tags=(tag,))
        self.tree.tag_configure('pos', foreground=GRN)
        self.tree.tag_configure('neg', foreground=RED)

        # ── matplotlib 圖表 ───────────────────────────────
        self.ax1.clear(); self.ax2.clear()
        self._style_axes()

        cum_inv  = [row['cum'] for row in rows]
        dds_pct  = [row['dd']*100 for row in rows]
        # 月資料點多，不顯示 marker；年資料顯示
        mk  = 'o' if r['ppy'] == 1 else ''
        mks = 4   if r['ppy'] == 1 else 0
        bw  = 0.7 if r['ppy'] == 1 else 1.0   # 月資料 bar 填滿

        # 上圖：市值 vs 累計投入
        self.ax1.fill_between(x_vals, cum_inv, alpha=0.18, color=ACC, label='累計投入')
        self.ax1.plot(x_vals, history, marker=mk, color=ACC,
                      linewidth=1.8, markersize=mks, label='組合市值')
        self.ax1.plot(x_vals, cum_inv, '--', color='#64748b', linewidth=1)
        self.ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        self.ax1.set_ylabel('金額', color=MUTED, fontsize=9)
        self.ax1.legend(loc='upper left', fontsize=8,
                        facecolor=SURF, edgecolor=BORD, labelcolor=TXT)
        self.ax1.set_title(
            f'{ticker}  {labels[0]}–{labels[-1]}  （按{unit}）\n'
            f'IRR={irr*100:.2f}%  CAGR={cagr*100:.2f}%  MaxDD={max_dd*100:.2f}% ({max_dd_lbl})',
            color=TXT, fontsize=9, pad=6)
        self.ax1.set_xticks(x_ticks)
        self.ax1.set_xticklabels(x_tlbls, rotation=45, ha='right', fontsize=7)

        # 下圖：回撤
        bar_cols = [RED if d < -0.01 else GRN for d in dds_pct]
        self.ax2.bar(x_vals, dds_pct, color=bar_cols, width=bw)
        self.ax2.axhline(0, color=BORD, linewidth=0.8)
        self.ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        self.ax2.set_ylabel('回撤 %', color=MUTED, fontsize=9)
        self.ax2.set_xlabel(x_lbl, color=MUTED, fontsize=9)
        self.ax2.set_xticks(x_ticks)
        self.ax2.set_xticklabels(x_tlbls, rotation=45, ha='right', fontsize=7)

        self.canvas.draw()

        self._set_status(
            f'✅  {ticker}  {labels[0]}–{labels[-1]}  ·  {n} 個{unit}  ·  '
            f'每{unit}投入 {fmt(inv_amt)}  ·  最終市值 {fmt(final_wealth)}  ·  '
            f'年化 IRR {irr*100:.2f}%', GRN)

    # ── 匯出 HTML 報告 ────────────────────────────────────
    def _on_export(self):
        if self._sim_result is None:
            messagebox.showinfo('提示', '請先執行模擬後再匯出報告')
            return
        threading.Thread(target=self._export_html, daemon=True).start()

    def _export_html(self):
        r = self._sim_result
        rows, labels = r['rows'], r['labels']
        embed = {
            'ticker': r['ticker'],
            'startYear': str(labels[0]), 'endYear': str(labels[-1]),
            'annInv': r['inv_amt'], 'years': [str(lb) for lb in labels],
            'rets':  [round(x['ret']*100, 4) for x in rows],
            'hist':  [round(x['val'], 2) for x in rows],
            'cum':   [round(x['cum'], 2) for x in rows],
            'dds':   [round(x['dd']*100, 4) for x in rows],
            'gains': [round(x['gain'], 2) for x in rows],
            'irr':   round(r['irr']*100, 4),
            'cagrP': round(r['cagr']*100, 4),
            'cagrI': round(r['cagr_i']*100, 4),
            'maxDD': round(r['max_dd']*100, 4),
            'maxDDY': str(r['max_dd_lbl']),
            'finalWealth':   round(r['final_wealth'], 2),
            'totalInvested': round(r['total_invested'], 2),
        }
        content = (HTML_TEMPLATE
                   .replace('__TICKER__', r['ticker'])
                   .replace('__START__',  str(labels[0]))
                   .replace('__END__',    str(labels[-1]))
                   .replace('__INV__',    f"{r['inv_amt']:,.0f}")
                   .replace('__DATA__',   json.dumps(embed, ensure_ascii=False)))

        fname = f"report_{r['ticker'].replace('.','_')}.html"
        out   = os.path.join(os.path.expanduser('~'), 'Downloads', fname)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(content)

        self._set_status(f'✅ HTML 報告已匯出：{out}', GRN)
        webbrowser.open(f"file:///{out.replace(os.sep, '/')}")

    def run(self):
        self.root.mainloop()


# ── 程式進入點 ────────────────────────────────────────────
if __name__ == '__main__':
    app = DCAApp()
    app.run()
