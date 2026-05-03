import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from research_lab.alpha_core import FractionalDifferencer, WaveletFeatureGenerator
from research_lab.alpha_labeler import AlphaLabeler

class MultiModalDataset(Dataset):
    """
    Encapsulates dual-stream data for Multi-Modal RankNet.
    """
    def __init__(self, x_seq, x_spatial, y, tickers=None, times=None):
        self.x_seq = x_seq
        self.x_spatial = x_spatial
        self.y = y
        self.tickers = tickers
        self.times = times

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x_seq': self.x_seq[idx],
            'x_spatial': self.x_spatial[idx],
            'y': self.y[idx]
        }

class AlphaUniverse:
    """
    AlphaUniverse: Orchestrates data fetching, rolling window generation,
    and target labeling for Multi-Modal Fusion (LSTM + ViT).
    """
    def __init__(self, data_engine: DataEngine = None):
        self.engine = data_engine or DataEngine()
        self.labeler = AlphaLabeler()

    def get_aligned_dataset(self, tickers: list, as_of_date: datetime, horizon: int = 21, lookback: int = 63, d_param: float = 0.4, scales: np.ndarray = None):
        """
        Returns a MultiModalDataset with sliding windows.
        
        Parameters:
            tickers: List of ticker symbols.
            as_of_date: Knowledge time threshold.
            horizon: Forward return horizon for labels.
            lookback: Sequence length (T).
            d_param: Fractional differentiation parameter.
            scales: Wavelet scales.
        """
        all_pit_views = {t: self.engine.get_pit_view(t, as_of_date) for t in tickers}
        
        fd = FractionalDifferencer(d=d_param)
        wfg = WaveletFeatureGenerator(scales=scales)
        
        x_seq_list = []
        x_spatial_list = []
        y_list = []
        ticker_list = []
        time_list = []
        
        # 1. Generate labels first to know alignment targets
        combined_pit_view = pd.concat(all_pit_views.values())
        returns_df = self.labeler.generate_labels(combined_pit_view, horizon=horizon)
        
        if 'SPY' in returns_df.columns:
            market_proxy = returns_df['SPY']
            asset_returns = returns_df.drop(columns=['SPY'])
        else:
            market_proxy = returns_df.mean(axis=1)
            asset_returns = returns_df
            
        residuals = self.labeler.residualize_universe(asset_returns, market_proxy)
        z_scored_labels = self.labeler.apply_z_score(residuals)
        
        # 2. Extract sequences for each ticker
        for ticker in tickers:
            if ticker not in z_scored_labels.columns:
                continue
                
            view = all_pit_views[ticker]
            if len(view) < lookback:
                continue
                
            # Apply FD
            stationary = fd.transform(view['close']).values
            # Apply Wavelets
            spectrogram = wfg.generate(pd.Series(stationary)) # Shape: (Scales, T_total)
            
            # Target labels for this ticker
            ticker_labels = z_scored_labels[ticker].dropna()
            
            # Align sequences ending at T with labels starting at T
            for t in range(lookback, len(view)):
                event_time = view.iloc[t-1]['event_time']
                
                if event_time not in ticker_labels.index:
                    continue
                
                # Sequential input: [T-lookback : T]
                seq_window = stationary[t-lookback:t].reshape(-1, 1)
                # Spatial input: [Scales, T-lookback : T]
                spatial_window = spectrogram[:, t-lookback:t]
                
                x_seq_list.append(seq_window)
                x_spatial_list.append(spatial_window)
                y_list.append(ticker_labels.loc[event_time])
                ticker_list.append(ticker)
                time_list.append(event_time)
                
        if not y_list:
            return None
            
        return MultiModalDataset(
            x_seq=torch.tensor(np.array(x_seq_list)).float(),
            x_spatial=torch.tensor(np.array(x_spatial_list)).unsqueeze(1).float(), # Add channel dim
            y=torch.tensor(np.array(y_list)).float(),
            tickers=ticker_list,
            times=time_list
        )
