import pandas as pd
import numpy as np

df = pd.DataFrame({
    'event_time': ['2020-01-01', '2020-01-01'],
    'ticker': ['AAPL', 'MSFT'],
    'close': [150.0, 200.0]
}).set_index('event_time')

# Suppose AAPL ffill produces 150.0, MSFT produces 200.0
# The groupby returns a Series with index ['2020-01-01', '2020-01-01']
res = df.groupby('ticker')['close'].ffill()

# Assign it back
df['close2'] = res

print(df)
