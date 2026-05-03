import pandas as pd
import numpy as np
from datetime import datetime
from research_lab.data_engine import DataEngine
from research_lab.alpha_core import FractionalDifferencer, WaveletFeatureGenerator
from research_lab.alpha_labeler import AlphaLabeler

class AlphaUniverse:
    """
    AlphaUniverse: A 'Deep Module' that orchestrates data fetching, 
    feature generation (Fractional Diff, Wavelets), and target labeling 
    (Residualization, Z-Scoring) into a single, bi-temporally aligned dataset.
    """
    def __init__(self, data_engine: DataEngine = None):
        self.engine = data_engine or DataEngine()
        self.labeler = AlphaLabeler()

    def get_aligned_dataset(self, tickers: list, as_of_date: datetime, horizon: int = 21, d_param: float = 0.4, scales: np.ndarray = None):
        """
        Returns a bi-temporally aligned DataFrame with features and labels.
        
        Parameters:
            tickers: List of ticker symbols.
            as_of_date: Knowledge time threshold for the PIT view.
            horizon: Forward return horizon for labels.
            d_param: Fractional differentiation parameter.
            scales: Wavelet scales.
            
        Returns:
            pd.DataFrame: A ready-for-training dataset with columns:
                          [event_time, ticker, scale_0, ..., scale_N, label]
        """
        # 1. Fetch PIT views for all tickers
        all_pit_views = {}
        for ticker in tickers:
            all_pit_views[ticker] = self.engine.get_pit_view(ticker, as_of_date)
            
        # 2. Generate features for each ticker
        all_features = []
        fd = FractionalDifferencer(d=d_param)
        wfg = WaveletFeatureGenerator(scales=scales)
        
        for ticker in tickers:
            view = all_pit_views[ticker]
            if view.empty:
                continue
            
            # Apply Fractional Differentiation
            stationary = fd.transform(view['close'])
            
            # Generate Wavelet Spectrogram
            spectrogram = wfg.generate(stationary) 
            
            # Convert spectrogram to features (n_timesteps x n_scales)
            feat_cols = [f'scale_{i}' for i in range(spectrogram.shape[0])]
            ticker_feats = pd.DataFrame(spectrogram.T, columns=feat_cols, index=view['event_time'])
            ticker_feats['ticker'] = ticker
            all_features.append(ticker_feats)
            
        if not all_features:
            return pd.DataFrame()
            
        features_df = pd.concat(all_features).reset_index()
        
        # 3. Generate labels (Residualized & Z-Scored)
        combined_pit_view = pd.concat(all_pit_views.values())
        
        # Generate forward returns
        returns_df = self.labeler.generate_labels(combined_pit_view, horizon=horizon)
        
        # Residualize Universe
        # Use 'SPY' as market proxy if available, else use cross-sectional mean
        if 'SPY' in returns_df.columns:
            market_proxy = returns_df['SPY']
            asset_returns = returns_df.drop(columns=['SPY'])
        else:
            market_proxy = returns_df.mean(axis=1)
            asset_returns = returns_df
            
        residuals = self.labeler.residualize_universe(asset_returns, market_proxy)
        z_scored_labels = self.labeler.apply_z_score(residuals)
        
        # Melt labels to long format: [event_time, ticker, label]
        labels_long = z_scored_labels.reset_index().melt(
            id_vars='event_time', 
            var_name='ticker', 
            value_name='label'
        )
        
        # 4. Bi-temporal alignment via inner join
        # This ensures that we only keep (event_time, ticker) pairs that have 
        # both features and a valid forward-looking label.
        final_dataset = pd.merge(
            features_df, 
            labels_long, 
            on=['event_time', 'ticker'], 
            how='inner'
        ).dropna(subset=['label'])
        
        return final_dataset.sort_values(['event_time', 'ticker'])
