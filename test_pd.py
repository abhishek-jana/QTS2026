import pandas as pd
import numpy as np

df = pd.DataFrame({
    'event_time': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
    'ticker': ['A', 'B', 'A', 'B'],
    'close': [1.0, np.nan, np.nan, 2.0]
}).set_index('event_time')

print(df)
try:
    df['close'] = df.groupby('ticker')['close'].ffill()
    print(df)
except Exception as e:
    print("Error:", e)
