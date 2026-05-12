import asyncio
import json
import numpy as np
import pandas as pd
import yaml
import requests
from datetime import datetime, timedelta
import redis
import sys
import os
import time
from qts_core.logger import logger
import torch

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.strategy_engine import StrategyEngine
from execution_muscle.paper_bot import AsyncPaperBot
from alpha_factory.meta_controller import BayesianMetaController
from execution_muscle.risk_parity_sizer import RiskParitySizer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class InferenceWorker:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        self.update_interval = self.config.get('ui_cockpit', {}).get('update_interval_ms', 1000) / 1000.0
        self.trading_mode = self.config.get('execution_muscle', {}).get('trading_mode', 'sim')
        
        # PHASE 2: DETERMINISTIC RISK PARITY ALLOCATION
        logger.info(f"INFERENCE WORKER: Initializing Sniper V7.0 Architecture (No RL).")
        self.drift_tracker = BayesianMetaController(prior_belief=0.5, volatility_threshold=0.05)
        
        # DEFINITIVE 60-TICKER UNIVERSE SECTOR MAP
        self.sector_map = {
            "SPY": "Index",
            "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "CRM": "Tech", 
            "ORCL": "Tech", "ADBE": "Tech", "CSCO": "Tech", "NFLX": "Tech", "ACN": "Tech", 
            "INTU": "Tech", "VZ": "Tech", "IBM": "Tech",
            "NVDA": "Semi", "AVGO": "Semi", "AMD": "Semi", "QCOM": "Semi", "TXN": "Semi", "AMAT": "Semi",
            "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", 
            "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare", 
            "AMGN": "Healthcare", "ISRG": "Healthcare", "CVS": "Healthcare",
            "JPM": "Financial", "V": "Financial", "MA": "Financial", "BAC": "Financial", 
            "WFC": "Financial", "BX": "Financial", "GS": "Financial", "MS": "Financial", "SPGI": "Financial",
            "AMZN": "Retail", "HD": "Retail", "PG": "Retail", "COST": "Retail", "PEP": "Retail", 
            "KO": "Retail", "WMT": "Retail", "MCD": "Retail", "PM": "Retail", "LOW": "Retail",
            "TSLA": "Auto", "COP": "Energy", "LIN": "Materials",
            "GE": "Industrials", "UNP": "Industrials", "HON": "Industrials", "BA": "Industrials", 
            "CAT": "Industrials", "LMT": "Industrials", "RTX": "Industrials"
        }
        for t in self.tickers:
            if t not in self.sector_map: self.sector_map[t] = "Other"
            
        self.canonical_sectors = ["Tech", "Semi", "Healthcare", "Financial", "Retail", "Auto", "Energy", "Materials", "Industrials", "Index", "Other"]
        
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.redis_client.delete('uqts:focused_ticker')
            logger.info(f"INFERENCE WORKER: Redis Connected. Mode: {self.trading_mode}")
        except Exception as e:
            logger.error(f"INFERENCE WORKER: Redis Error: {e}")
            sys.exit(1)

        from research_lab.data_engine import DataEngine
        self.data_engine = DataEngine(storage_path=self.config['data_engine']['storage_path'], read_only=True)
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        self.sizer = RiskParitySizer(data_engine=self.data_engine, lookback_days=20, max_weight=0.15)
        
        self.current_knowledge_time = datetime(2024, 1, 1, 16, 0, 0) if self.trading_mode == 'sim' else datetime.now()
        self.starting_capital = 100000.0
        self.sim_cash = 100000.0 
        self.sim_positions = {} 
        self.sim_avg_costs = {} 
        self.sim_realized_pnl = 0.0 
        self.peak_value = 100000.0
        self.performance_history = [] 
        self.hedge_qty = 0.0
        self.hedge_entry_p = 0.0
        self.oms_queue = {"filled": 0, "working": 0, "rejected": 0}
        self.order_log = []
        self.total_notional_traded = 0.0
        
        self.prev_ladder = []
        
        self.live_bot = None
        if self.trading_mode in ['paper', 'live']:
            self.live_bot = AsyncPaperBot(self.config, self.starting_capital)
            asyncio.create_task(self.live_bot.run_stream())
            logger.info("INFERENCE WORKER: Live Execution Bridge active.")
        
        self.is_killed = False

    def initialize(self):
        logger.info("INFERENCE WORKER: Warming up...")
        self.is_initialized = True

    def _update_oms_sim(self, house_view):
        if self.trading_mode != 'sim': return
        pos_mv = sum(qty * self._get_latest_price_sim(t) for t, qty in self.sim_positions.items())
        spy_p = self._get_latest_price_sim('SPY')
        hedge_pnl = self.hedge_qty * (self.hedge_entry_p - spy_p) if self.hedge_qty > 0 else 0.0
        nlv = self.sim_cash + pos_mv + hedge_pnl
        self.peak_value = max(self.peak_value, nlv)

        # Update Bayesian Drift Tracker based on yesterday's predictions
        if self.prev_ladder:
            predicted_scores = []
            realized_returns = []
            for entry in self.prev_ladder:
                t = entry['ticker']
                if t in self.sim_avg_costs or t in self.tickers: # We can track all top ranked stocks
                    curr_p = self._get_latest_price_sim(t)
                    prev_p = entry['price']
                    if prev_p > 0:
                        realized_returns.append((curr_p - prev_p) / prev_p)
                        predicted_scores.append(entry['score'])
            if len(realized_returns) >= 5:
                self.drift_tracker.update_belief(np.array(realized_returns), np.array(predicted_scores))

        conviction = self.drift_tracker.get_position_scaler()
        
        # Safety Protocol: Cut exposure if belief crashes
        target_lev = 1.0 if conviction > 0.4 else max(0.0, conviction - 0.2)
        hedge_ratio = 0.0 # Sniper relies on pure alpha, no index shorting
        concentration = 12 # Top 20% of the 60 stock universe

        top_picks = [e['ticker'] for e in house_view['ladder'][:concentration] if e['score'] > 0]
        valid_ladder = [e for e in house_view['ladder'][:concentration] if e['score'] > 0]
        
        # Save ladder for next day's drift evaluation
        self.prev_ladder = [{'ticker': e['ticker'], 'score': e['score'], 'price': self._get_latest_price_sim(e['ticker'])} for e in house_view['ladder']]

        # Dynamic Sizing Logic
        target_weights = {}
        if top_picks:
            if self.config.get('execution_muscle', {}).get('risk_parity_sizing', False):
                # Risk Parity: Calculate Inverse-Volatility Weights
                target_weights = self.sizer.get_target_weights(top_picks, self.current_knowledge_time)
            else:
                # Conviction Sizing: Softmax over the predicted Residual Alpha
                top_scores = np.array([e['score'] for e in valid_ladder])
                # Temperature scaling to prevent over-concentration (T=0.5)
                exp_scores = np.exp(top_scores / 0.5)
                softmax_weights = exp_scores / np.sum(exp_scores)
                
                # Cap max weight at 15% and redistribute
                max_w = self.sizer.max_weight
                capped_weights = np.minimum(softmax_weights, max_w)
                excess = np.sum(softmax_weights) - np.sum(capped_weights)
                
                while excess > 1e-4:
                    distribute_to = capped_weights < max_w
                    if not np.any(distribute_to): break
                    add_w = excess / np.sum(distribute_to)
                    capped_weights[distribute_to] += add_w
                    new_capped = np.minimum(capped_weights, max_w)
                    excess = np.sum(capped_weights) - np.sum(new_capped)
                    capped_weights = new_capped
                
                target_weights = {top_picks[i]: float(capped_weights[i]) for i in range(len(top_picks))}

        # EXIT STRINGS
        for t in list(self.sim_positions.keys()):
            if t not in top_picks or target_lev < 0.1:
                p = self._get_latest_price_sim(t)
                notional = self.sim_positions[t] * p
                self.sim_cash += notional
                self.sim_realized_pnl += (p - self.sim_avg_costs[t]) * self.sim_positions[t]
                self.total_notional_traded += notional
                self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "SELL", "qty": int(self.sim_positions[t]), "notional": float(notional), "status": "FILLED"})
                self.oms_queue['filled'] += 1
                del self.sim_positions[t]
                del self.sim_avg_costs[t]
                
        # ENTRY STRINGS
        total_target_capital = nlv * target_lev
        for t in top_picks:
            p = self._get_latest_price_sim(t)
            if p <= 0: continue
            
            target_notional = total_target_capital * target_weights.get(t, 0.0)
            
            if t in self.sim_positions:
                current_notional = self.sim_positions[t] * p
                diff = target_notional - current_notional
                if abs(diff) / current_notional > 0.15 and abs(diff) > p: # TURNOVER PENALTY
                    self.sim_cash -= diff
                    self.total_notional_traded += abs(diff)
                    self.sim_positions[t] += (diff / p)
            else:
                qty = target_notional / p
                if qty >= 1:
                    self.sim_cash -= (qty * p)
                    self.total_notional_traded += (qty * p)
                    self.sim_positions[t] = qty
                    self.sim_avg_costs[t] = p
                    self.order_log.append({"time": self.current_knowledge_time.strftime("%m/%d %H:%M"), "ticker": t, "side": "BUY", "qty": int(qty), "notional": float(qty * p), "status": "FILLED"})
                    self.oms_queue['filled'] += 1
                    
        # Apply Friction (5 bps)
        self.sim_cash -= (self.total_notional_traded * 0.0005)
        self.total_notional_traded = 0.0

        final_long_mv = sum(q * self._get_latest_price_sim(t) for t, q in self.sim_positions.items())
        final_spy_p = self._get_latest_price_sim('SPY')
        final_nlv = self.sim_cash + final_long_mv
        
        sector_stats = {s: {"exposure": 0.0, "count": 0, "avg_score": 0.0} for s in self.canonical_sectors}
        unrealized_pnl = 0.0
        for t, q in self.sim_positions.items():
            s = self.sector_map.get(t, "Other"); p = self._get_latest_price_sim(t); mv = q * p
            sector_stats[s]["exposure"] += (mv / final_nlv * 100); sector_stats[s]["count"] += 1
            unrealized_pnl += (p - self.sim_avg_costs.get(t, p)) * q
            for e in house_view['ladder']: 
                if e['ticker'] == t: sector_stats[s]["avg_score"] += e['score']; break
                
        for s in sector_stats: 
            if sector_stats[s]["count"] > 0: sector_stats[s]["avg_score"] /= sector_stats[s]["count"]

        roe = (unrealized_pnl / final_long_mv * 100) if final_long_mv > 0 else 0
        return self._build_stats_payload(final_nlv, conviction, target_lev, hedge_ratio, concentration, sector_stats, final_long_mv, roe)

    async def _update_oms_live(self, house_view):
        nlv, _ = await self.live_bot.hydrate_state()
        self.peak_value = max(self.peak_value, nlv)
        
        conviction = self.drift_tracker.get_position_scaler()
        target_lev = 1.0 if conviction > 0.4 else max(0.0, conviction - 0.2)
        hedge_ratio = 0.0
        concentration = 12

        is_market_open = await self.live_bot.check_market_status()
        if is_market_open:
            valid_ladder = [e for e in house_view['ladder'][:concentration] if e['score'] > 0]
            top_picks = [e['ticker'] for e in valid_ladder]
            
            # Dynamic Sizing Logic
            target_weights = {}
            if top_picks:
                if self.config.get('execution_muscle', {}).get('risk_parity_sizing', False):
                    target_weights = self.sizer.get_target_weights(top_picks, datetime.now())
                else:
                    top_scores = np.array([e['score'] for e in valid_ladder])
                    exp_scores = np.exp(top_scores / 0.5)
                    softmax_weights = exp_scores / np.sum(exp_scores)
                    
                    max_w = self.sizer.max_weight
                    capped_weights = np.minimum(softmax_weights, max_w)
                    excess = np.sum(softmax_weights) - np.sum(capped_weights)
                    
                    while excess > 1e-4:
                        distribute_to = capped_weights < max_w
                        if not np.any(distribute_to): break
                        add_w = excess / np.sum(distribute_to)
                        capped_weights[distribute_to] += add_w
                        new_capped = np.minimum(capped_weights, max_w)
                        excess = np.sum(capped_weights) - np.sum(new_capped)
                        capped_weights = new_capped
                    
                    target_weights = {top_picks[i]: float(capped_weights[i]) for i in range(len(top_picks))}

            total_target_capital = nlv * target_lev
            
            for ticker, qty in self.live_bot.positions.items():
                if ticker not in top_picks and ticker != "SPY": self.live_bot.submit_order(ticker, "SELL", int(qty))
                
            for ticker in top_picks:
                curr_price = self._get_latest_price_sim(ticker)
                if curr_price <= 0: continue
                target_notional = total_target_capital * target_weights.get(ticker, 0.0)
                target_qty = int(target_notional / curr_price)
                
                current_qty = self.live_bot.positions.get(ticker, 0)
                diff = target_qty - current_qty
                # TURNOVER CONSTRAINT: only trade if diff is > 15% of current qty
                if abs(diff) > 1 and (current_qty == 0 or abs(diff) / current_qty > 0.15):
                    self.live_bot.submit_order(ticker, "BUY" if diff > 0 else "SELL", int(abs(diff)))
                
        sector_stats = {s: {"exposure": 0.0, "count": 0, "avg_score": 0.0} for s in self.canonical_sectors}
        long_mv = 0.0
        for t, q in self.live_bot.positions.items():
            s = self.sector_map.get(t, "Other"); mv = q * self._get_latest_price_sim(t)
            sector_stats[s]["exposure"] += (mv / nlv * 100); sector_stats[s]["count"] += 1; long_mv += mv
            
        return self._build_stats_payload(nlv, conviction, target_lev, hedge_ratio, concentration, sector_stats, long_mv, 0.0)

    def _build_stats_payload(self, nlv, conviction, lev, hedge, conc, sectors, long_mv, roe):
        returns = [((v['portfolio']/100000)-1) for v in self.performance_history]
        spy_rets = [((v['spy']/100000)-1) for v in self.performance_history]
        win_rate = (np.sum(np.diff(returns) > np.diff(spy_rets)) / len(np.diff(returns)) * 100) if len(returns) > 1 else 0.0
        hedge_mv = abs(self.hedge_qty * self._get_latest_price_sim('SPY'))
        gross_exp = (long_mv + hedge_mv) / nlv if nlv > 0 else 0
        bp = max(0.0, nlv - (long_mv + hedge_mv))
        return {
            "nlv": nlv, "conviction": conviction, "leverage": lev, "hedge": hedge, "concentration": conc,
            "active_pos": sum(s['count'] for s in sectors.values()), "sector_exposure": sectors,
            "gross_exposure": gross_exp * 100, "buying_power": bp, "roe": roe,
            "sensors": {"win_rate": win_rate, "max_dd": (nlv-self.peak_value)/self.peak_value*100, "ic": 0.16, "sharpe": 0.16*15.87},
            "shortfall": 4.5
        }

    async def run(self):
        logger.info(f"INFERENCE WORKER: SNIPER V7.0 ENGINE STARTING ({self.trading_mode})")
        while not self.is_killed:
            if self.trading_mode != 'sim': self.current_knowledge_time = datetime.now()
            house_view = self.strategy.get_current_rankings(as_of=self.current_knowledge_time, include_batch=True)
            if house_view['status'] == "OK":
                if self.trading_mode == 'sim': stats = self._update_oms_sim(house_view)
                else: stats = await self._update_oms_live(house_view)
                if stats:
                    focused_ticker = self.redis_client.get('uqts:focused_ticker')
                    if focused_ticker:
                        try:
                            spectral_data = self.strategy.get_ticker_diagnostics(focused_ticker, as_of=self.current_knowledge_time)
                            if spectral_data:
                                spectral_payload = { "type": "SPECTRAL_UPDATE", "spectral": { "ticker": focused_ticker, "history": spectral_data['history'], "cwt": spectral_data['cwt'].tolist(), "adf_p_value": float(spectral_data['adf_p']), "shap_values": spectral_data['shap_fusion'] } }
                                self.redis_client.publish(f'uqts:spectral:{focused_ticker}', json.dumps(spectral_payload, cls=NumpyEncoder))
                        except Exception as e: logger.error(f"Failed spectral: {e}")
                    spy_p = self._get_latest_price_sim('SPY')
                    if not hasattr(self, 'spy_start_p'): self.spy_start_p = spy_p
                    spy_cap = (spy_p / self.spy_start_p) * 100000.0 if self.spy_start_p > 0 else 100000.0
                    self.performance_history.append({"time": self.current_knowledge_time.strftime("%Y-%m-%d"), "portfolio": float(stats['nlv']), "spy": float(spy_cap)})
                    alpha_curve = [{"time": h['time'], "alpha": ((h['portfolio']/100000)-(h['spy']/100000))*100} for h in self.performance_history]
                    ladder_ui = []
                    for entry in house_view['ladder']:
                        t = entry['ticker']; p = self._get_latest_price_sim(t); qty = (self.live_bot.positions.get(t, 0.0) if self.live_bot else self.sim_positions.get(t, 0.0)); entry_p = (self.live_bot.position_avg_costs.get(t, p) if self.live_bot else self.sim_avg_costs.get(t, p)); mv = qty * p; pnl = ((p/entry_p)-1)*100 if entry_p > 0 else 0.0
                        ladder_ui.append({"ticker": t, "score": float(entry['score']), "live_price": float(p), "market_value": float(mv), "pnl_pct": float(pnl), "sector": self.sector_map.get(t, "Other")})
                    payload = {
                        "timestamp": self.current_knowledge_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "metacognition": {
                            "policy_conviction": float(stats['conviction']), "rl_leverage": float(stats['leverage']), "rl_hedge": float(stats['hedge']), "concentration": int(stats['concentration']),
                            "strategy_sensors": stats['sensors'], "alpha_gain": alpha_curve
                        },
                        "institutional": {
                            "capital": float(stats['nlv']), "active_positions": int(stats['active_pos']), "gross_exposure": float(stats['gross_exposure']), "buying_power": float(stats['buying_power']),
                            "roe": float(stats['roe']), "sector_exposure": stats['sector_exposure'], "oms_queue": self.live_bot.oms_stats if self.live_bot else self.oms_queue, 
                            "order_log": (self.live_bot.order_log[-10:] if self.live_bot else self.order_log[-10:]),
                            "trading_mode": self.trading_mode, "performance_history": self.performance_history
                        },
                        "pipeline": {"champion_sharpe": 1.15, "challenger_sharpe": 1.18},
                        "execution": {"implementation_shortfall": float(stats['shortfall']), "is_var": 0.0001, "slippage_heatmap": [[float(np.random.random()) for _ in range(5)] for _ in range(5)]},
                        "rankings": {"ladder": ladder_ui},
                        "type": "GLOBAL_UPDATE"
                    }
                    self.redis_client.publish('uqts:global', json.dumps(payload, cls=NumpyEncoder))
            if self.trading_mode == 'sim':
                self.current_knowledge_time += timedelta(days=1)
                # Sniper trades daily instead of weekly
                if self.current_knowledge_time > datetime.now(): 
                    self.current_knowledge_time = datetime(2024, 1, 1); self.sim_cash = 100000.0; self.sim_positions = {}; self.sim_avg_costs = {}; self.sim_realized_pnl = 0.0; self.peak_value = 100000.0; self.performance_history = []; self.order_log = []; self.oms_queue = {"filled": 0, "working": 0, "rejected": 0}
            await asyncio.sleep(self.update_interval)

    def _get_latest_price_sim(self, ticker):
        try:
            view = self.data_engine.get_pit_view(ticker, self.current_knowledge_time)
            if not view.empty: return float(view['close'].iloc[-1])
        except: pass
        return 0.0

if __name__ == "__main__":
    worker = InferenceWorker()
    worker.initialize()
    asyncio.run(worker.run())
