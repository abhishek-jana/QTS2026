import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine
from research_lab.alpha_ranker import RankNet
import torch

class WFOEngine:
    """
    Walk-Forward Optimization Engine.
    Automates model retraining while strictly honoring Knowledge Time.
    """
    def __init__(self, data_engine: DataEngine, model_params: dict):
        self.data_engine = data_engine
        self.model_params = model_params
        self.models = {} # Store models by date: {date: model_state}

    def train_step(self, train_view: pd.DataFrame, target_labels: pd.DataFrame):
        """
        Trains a single RankNet instance on the provided PIT view.
        Uses the high-level .fit() interface.
        """
        input_dim = self.model_params['input_dim']
        model = RankNet(input_dim)
        
        # Convert DataFrames to Tensors
        X = torch.tensor(train_view.values).float()
        y = torch.tensor(target_labels.values).float()
        
        # High-level training call
        model.fit(X, y, epochs=self.model_params.get('epochs', 50))
        
        return model.state_dict()

    def run_pipeline(self, start_date: datetime, end_date: datetime, step_days: int = 30):
        """
        Iterates through time, retraining the model at each step.
        """
        current_date = start_date
        while current_date <= end_date:
            # 1. Get PIT view for training
            # Use all data known as of current_date
            train_view = self.data_engine.registry[
                self.data_engine.registry['knowledge_time'] <= current_date
            ]
            
            # 2. Train model
            # state = self.train_step(train_view, ...)
            # self.models[current_date] = state
            
            print(f"WFO Retrain triggered for Knowledge Time: {current_date}")
            
            current_date += timedelta(days=step_days)
        
        return self.models
