import pandas as pd
from pandas.tseries.offsets import BDay

# 读取数据
df = pd.read_csv('macro_economy.csv')

# 将 date 列转换为日期时间格式
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# 处理包含逗号的字符串列
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].replace(',', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 按日期排序并重置索引
df = df.sort_values('date').reset_index(drop=True)

# 按每周/每月/每季度/每年的最后一天数据填充缺失值
for freq in ['W', 'ME', 'QE', 'YE']:
    fill_values = df.groupby(pd.Grouper(key='date', freq=freq)).transform('last')
    df = df.fillna(fill_values)

# 筛选 2014-01-01 至 2016-01-01 期间的数据
filtered_df = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2016-01-01')]

# 筛选出交易日的数据
filtered_df = filtered_df[filtered_df['date'].apply(lambda x: x.isoweekday() <= 5)]

# 或者使用 BDay 判断（需要 pandas.tseries.offsets 模块）
# is_business_day = pd.Series(filtered_df['date'].map(lambda x: BDay().onOffset(x)))
# filtered_df = filtered_df[is_business_day.values]

# 将结果保存为 CSV 文件
csv_path = 'filled_macro_economy.csv'
filtered_df.to_csv(csv_path, index=False)