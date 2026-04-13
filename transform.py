# 未來對接全市場清單的寫法：
df_all_stocks = pd.read_csv('tw_stocks.csv')

# 將 DataFrame 轉為字典： {'2330': ['台積電'], '2317': ['鴻海']}
full_market_dict = {}
for index, row in df_all_stocks.iterrows():
    # 將公司全名與簡稱都放進去增加命中率，例如把 "台灣積體電路" 跟 "台積電" 都當作 tag
    full_market_dict[str(row['stock_id'])] = [row['stock_name']] 
    
# 將這個 full_market_dict 傳入 fetch_and_process_rss 即可！