import pandas as pd
import numpy as np

df = pd.DataFrame({
    'event_time': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
    'ticker': ['A', 'B', 'B', 'A'],
    'close': [1.0, np.nan, 2.0, np.nan]
}).set_index('event_time')

print("Original:")
print(df)
df['close'] = df.groupby('ticker')['close'].ffill()
print("\nAfter groupby.ffill (non-unique index):")
print(df)
