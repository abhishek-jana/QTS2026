import React, { useState, useEffect, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import { 
  Activity, 
  ShieldAlert, 
  TrendingUp, 
  Zap, 
  Terminal, 
  Cpu, 
  Gauge, 
  AlertTriangle,
  HelpCircle,
  X
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter
} from 'recharts';

// --- Components ---

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ error, errorInfo });
    console.error("FATAL UI ERROR:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-full bg-slate-900 text-red-500 p-10 font-mono overflow-auto">
          <h1 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <ShieldAlert className="w-8 h-8" /> 
            CORE SYSTEM CRASH DETECTED
          </h1>
          <div className="bg-black p-6 border border-red-500/30 rounded">
            <div className="text-red-400 font-bold mb-2 uppercase text-xs tracking-widest">Error Stack Trace:</div>
            <pre className="text-[10px] leading-relaxed whitespace-pre-wrap">
              {this.state.error && this.state.error.toString()}
              {this.state.errorInfo && this.state.errorInfo.componentStack}
            </pre>
          </div>
          <button 
            onClick={() => window.location.reload()}
            className="mt-6 bg-red-600 text-white px-4 py-2 font-bold uppercase text-xs hover:bg-red-500 transition-colors"
          >
            Reboot Cockpit
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const PriceChart = ({ data, ticker }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const seriesRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      width: chartContainerRef.current.clientWidth || 400,
      height: 250,
      timeScale: {
        borderColor: '#1e293b',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);
    
    // Initial data load if available
    if (data && data.length > 0) {
      try {
        // Deduplicate and sort data by time to prevent lightweight-charts crash
        const seen = new Set();
        const cleanData = data
          .filter(d => {
            if (seen.has(d.time)) return false;
            seen.add(d.time);
            return true;
          })
          .sort((a, b) => a.time - b.time);
        
        series.setData(cleanData);
      } catch (e) {
        console.error("Initial chart data load error:", e);
      }
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, []); // Only run once on mount

  useEffect(() => {
    if (seriesRef.current && data && data.length > 0) {
      try {
        const seen = new Set();
        const cleanData = data
          .filter(d => {
            if (seen.has(d.time)) return false;
            seen.add(d.time);
            return true;
          })
          .sort((a, b) => a.time - b.time);

        seriesRef.current.setData(cleanData);
      } catch (e) {
        console.error("Chart data update error:", e);
      }
    }
  }, [data]);

  return (
    <div className="w-full flex flex-col gap-1">
      <div className="text-[10px] font-mono text-emerald-500 uppercase font-bold tracking-tighter">
        {ticker} LIVE FEED (CANVAS_RENDERED)
      </div>
      <div ref={chartContainerRef} className="w-full h-[250px] border border-slate-800 bg-black/50" />
    </div>
  );
};

const Panel = ({ title, icon: Icon, children, className = "" }) => (
  <div className={`bg-slate-900 border border-slate-700 p-4 flex flex-col h-fit ${className}`}>
    <div className="flex-none flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
      <Icon className="w-4 h-4 text-emerald-500" />
      <h2 className="text-xs font-mono uppercase tracking-widest text-slate-300 font-bold">{title}</h2>
    </div>
    <div className="flex-1">
      {children}
    </div>
  </div>
);


const Heatmap = ({ data, title }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!data || !data.length || !data[0] || !data[0].length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rows = data.length;
    const cols = data[0].length;
    
    // DRAW resolution
    canvas.width = cols;
    canvas.height = rows;
    
    const imageData = ctx.createImageData(cols, rows);
    const flatData = data.flat();
    const maxVal = Math.max(...flatData.slice(0, 5000), 0.001);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = data[i][j];
        const ratio = Math.min(1.0, val / maxVal);
        const idx = (i * cols + j) * 4;

        // Inferno
        imageData.data[idx] = Math.min(255, ratio * 400); 
        imageData.data[idx + 1] = Math.max(0, (ratio - 0.3) * 500); 
        imageData.data[idx + 2] = Math.max(0, (ratio - 0.6) * 600); 
        imageData.data[idx + 3] = 255; 
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <div className="w-full h-full flex flex-col min-h-0">
      <div className="text-[10px] font-mono text-slate-500 mb-1 tracking-tighter truncate">{title}</div>
      <div className="flex-1 min-h-0 relative border border-slate-800 bg-black">
        <canvas 
          ref={canvasRef} 
          className="absolute inset-0 w-full h-full" 
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
    </div>
  );
};

const RankingGrid = ({ ladder, onSelectTicker }) => {
  // Unique-ify the list. One entry per ticker, highest score.
  const aggregated = React.useMemo(() => {
    const map = {};
    ladder.forEach(item => {
      if (!map[item.ticker] || item.score > map[item.ticker].score) {
        map[item.ticker] = { score: item.score, price: item.live_price };
      }
    });
    return Object.entries(map)
      .map(([ticker, data]) => ({ ticker, score: data.score, price: data.price }))
      .sort((a, b) => b.score - a.score);
  }, [ladder]);

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar pr-2">
      <div className="text-[10px] font-bold text-emerald-500/80 mb-2 uppercase border-b border-emerald-900/30 pb-1">Decile Ladder (House View)</div>
      <table className="w-full text-[10px] border-collapse">
        <thead className="sticky top-0 bg-slate-900 shadow-sm">
          <tr>
            <th className="text-left py-2 text-slate-500 uppercase font-normal">Ticker</th>
            <th className="text-right py-2 text-slate-500 uppercase font-normal pr-4">Live Price</th>
            <th className="text-right py-2 text-slate-500 uppercase font-normal pr-4">Z-Score</th>
            <th className="text-right py-2 text-slate-500 uppercase font-normal">Action</th>
          </tr>
        </thead>
        <tbody>
          {aggregated.map((row, i) => (
            <tr 
              key={row.ticker} 
              onClick={() => onSelectTicker(row.ticker)}
              className="border-b border-slate-800/40 group hover:bg-emerald-500/20 cursor-pointer transition-colors"
            >
              <td className="py-2.5 flex items-center gap-2">
                <div className={`w-1 h-3 ${row.score > 0 ? 'bg-emerald-500' : 'bg-red-500'} shadow-sm`} />
                <span className="font-bold text-slate-200 tracking-tight group-hover:text-emerald-400">{row.ticker}</span>
              </td>
              <td className="text-right py-2.5 font-mono text-slate-400 pr-4">
                ${typeof row.price === 'number' ? row.price.toFixed(2) : "---"}
              </td>
              <td className={`text-right py-2.5 font-mono ${typeof row.score === 'number' && row.score > 0 ? 'text-emerald-400' : 'text-red-400'} pr-4`}>
                {typeof row.score === 'number' ? row.score.toFixed(4) : "---"}
              </td>
              <td className="text-right py-2.5">
                <span className={`text-[8px] px-1 border ${row.score > 0 ? 'border-emerald-800 text-emerald-600' : 'border-red-900 text-red-700'} uppercase font-bold`}>
                  {row.score > 0 ? 'Long' : 'Short'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// --- Main App ---

const MissionManual = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const sections = [
    {
      title: "1. Spectral & Signal Viewer",
      items: [
        {
          label: "Price Chart (Top)",
          detail: "HOW TO READ: Represents raw market fluid. Candles show OHLCV. Green = Price up, Red = Price down. Synchronized to Simulation Knowledge Time (Edge of Time)."
        },
        {
          label: "Wavelet Spectrogram (Center Heatmap)",
          detail: "HOW TO READ: Y-axis represents 'Frequency Scales'. Top rows are High-Resolution (Fast cycles/Noise), Bottom rows are Low-Resolution (Slow cycles/Trends). Color brightness = ENERGY. Bright spots at the top indicate high-frequency volatility; bright spots at the bottom indicate long-term structural momentum."
        }
      ]
    },
    {
      title: "2. Statistical Integrity",
      items: [
        {
          label: "ADF p-value",
          detail: "HOW TO READ: Numerical stability indicator. MUST be below 0.05. If the number is flashing RED, the signal is 'Non-Stationary' (wandering) and predictions should be discounted."
        },
        {
          label: "Model State Labels",
          detail: "HOW TO READ: READY (Green) = Model is active. STATIONARY (Green) = Fractional differentiation (d=0.4) is correctly configured to remove noise while preserving memory."
        }
      ]
    },
    {
      title: "3. Feature Importance (SHAP)",
      items: [
        {
          label: "SHAP Bar Chart",
          detail: "HOW TO READ: X-axis is 'Contribution Power'. LONGER BARS = Higher influence. It tells you 'WHY' a stock is ranked high/low. (e.g., If Momentum is the longest bar, the model is chasing the recent trend)."
        }
      ]
    },
    {
      title: "4. Execution & Reality Check",
      items: [
        {
          label: "Implementation Shortfall (IS)",
          detail: "HOW TO READ: Execution efficiency in BPS (0.01%). Lower is better. If IS > 5.0, the model's trades are losing too much money to market impact or poor liquidity."
        },
        {
          label: "OMS Queue & Live Log",
          detail: "HOW TO READ: Real-time trade monitoring. F = Filled, W = Working, R = Rejected. The log shows instant feedback on every trade attempt from the Execution Muscle."
        },
        {
          label: "Slippage Heatmap",
          detail: "HOW TO READ: 5x5 liquidity distribution. Darker cells = Thin Liquidity, Brighter cells = Deep Liquidity. Dark right-most columns indicate high buy-side slippage risk."
        }
      ]
    },
    {
      title: "5. Metacognition Panel",
      items: [
        {
          label: "Bayesian Belief Score (Gauge)",
          detail: "HOW TO READ: The system's 'Confidence' in its own model. 80%+ is High Conviction. < 60% triggers automatic risk scaling. It 'learns' by comparing predicted ranks to actual future returns."
        },
        {
          label: "Manifold Drift (Scatter)",
          detail: "HOW TO READ: Distance = Divergence. Gray dots are the model's 'Training Base'. The Green cross is the 'Current Live State'. If the cross drifts far from the gray cluster, the market regime has changed (Concept Drift)."
        },
        {
          label: "Alpha Decay (Line)",
          detail: "HOW TO READ: Cumulative Information Gain. UPWARD SLOPE = Model is extracting value. NOTE: This curve only 'ticks' when new daily market data is realized (every 24 simulation hours). Flatlines between these jumps are normal behavior."
        }
      ]
    },
    {
      title: "6. Ranking Grid & Risk",
      items: [
        {
          label: "Decile Ladder",
          detail: "HOW TO READ: Sorted prediction ladder. Top 25% are high-conviction LONGS (predicted winners), Bottom 25% are high-conviction SHORTS (predicted losers)."
        },
        {
          label: "Sector Exposure Matrix",
          detail: "HOW TO READ: Aggregates risk by industry weighted by model conviction (Score * Belief). If Tech is +20.0%, you have a heavy long bias. We aim for 'Sector Neutral'."
        },

      ]
    },
    {
      title: "7. Systems Ops (Header)",
      items: [
        {
          label: "Gross & Net Exposure",
          detail: "HOW TO READ: Gross = Total market leverage. Net = Portfolio direction. A Net Exp of 0% means the portfolio is perfectly 'Market Neutral' (Longs = Shorts)."
        },
        {
          label: "API Latency & Freshness",
          detail: "HOW TO READ: API Latency = Time for broker response (Normal < 300ms). Data Freshness = Time since last market poll. Because we use 1-minute bars, freshness will naturally cycle from 1s to 60s."
        },

      ]
    },
    {
      title: "8. Research Pipeline Control",
      items: [
        {
          label: "Champion vs Challenger",
          detail: "HOW TO READ: Real-time A/B test. Sharpe Ratio measures risk-adjusted return. If the 'Challenger' (training on newer data) consistently beats the 'Champion', it is ready for deployment."
        }
      ]
    }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md p-4">
      <div className="bg-slate-900 border border-emerald-500/50 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col shadow-[0_0_60px_rgba(16,185,129,0.3)]">
        <div className="flex-none p-6 border-b border-slate-800 flex justify-between items-center bg-emerald-500/5">
          <div className="flex items-center gap-3">
            <Terminal className="w-6 h-6 text-emerald-500" />
            <h2 className="text-2xl font-bold text-emerald-400 tracking-tighter uppercase">UQTS-2026 Operator Manual</h2>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors bg-slate-800/50 p-2 rounded-full">
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-8 space-y-10 no-scrollbar">
          {sections.map((section, idx) => (
            <div key={idx} className="space-y-4">
              <h3 className="text-emerald-500 font-bold uppercase text-sm tracking-[0.2em] border-b border-emerald-900/30 pb-2">
                {section.title}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {section.items.map((item, i) => (
                  <div key={i} className="bg-black/30 p-4 border border-slate-800/50 hover:border-emerald-500/20 transition-colors">
                    <h4 className="text-slate-100 font-bold text-[10px] uppercase tracking-wider mb-2 flex items-center gap-2">
                      <div className="w-1 h-2 bg-emerald-500" /> {item.label}
                    </h4>
                    <p className="text-slate-400 leading-relaxed text-[11px] font-mono">
                      {item.detail}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ))}
          
          <div className="pt-8 border-t border-slate-800">
            <div className="bg-emerald-900/10 p-6 border border-emerald-900/30 rounded-sm">
              <div className="text-[10px] text-emerald-500 font-bold uppercase mb-2">Core Philosophy</div>
              <div className="text-[11px] text-emerald-700 font-mono italic leading-relaxed">
                "We do not predict prices; we rank the multi-resolution energy states of the market fluid. Stationarity is our requirement. Relative outperformance is our signal. Bayesian belief is our safety."
              </div>
            </div>
          </div>
        </div>

        <div className="flex-none p-6 bg-black/40 border-t border-slate-800 flex justify-center items-center gap-4">
          <div className="text-[10px] text-slate-600 font-mono uppercase">Status: Authorization Required</div>
          <button 
            onClick={onClose}
            className="bg-emerald-600 text-black px-12 py-3 font-bold uppercase text-xs hover:bg-emerald-400 transition-all shadow-[0_0_20px_rgba(16,185,129,0.2)] active:scale-95"
          >
            Acknowledge Mission Specs
          </button>
        </div>
      </div>
    </div>
  );
};

export default function MissionControl() {
  const [globalData, setGlobalData] = useState(null);
  const [spectralData, setSpectralData] = useState(null);
  const [status, setStatus] = useState('connecting');
  const [showManual, setShowManual] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [isKilled, setIsKilled] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/cockpit');
    
    ws.current.onopen = () => {
      console.log("📡 WebSocket Connected");
      setStatus('active');
    };
    ws.current.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        console.log(`📥 INCOMING [${payload.type}]`, payload);
        if (payload.type === 'GLOBAL_UPDATE') {
          setGlobalData(payload);
        } else if (payload.type === 'SPECTRAL_UPDATE') {
          setSpectralData(payload.spectral);
        } else if (payload.type === 'ALERT') {
          setAlerts(prev => [...prev, { id: Date.now(), msg: payload.msg }]);
          setIsKilled(true);
        }
      } catch (e) {
        console.error("Payload parse error:", e);
      }
    };
    ws.current.onclose = () => {
      console.log("📡 WebSocket Disconnected");
      setStatus('disconnected');
    };

    return () => ws.current.close();
  }, []);

  const triggerKillSwitch = () => {
    if (ws.current) ws.current.send('KILL_SWITCH');
  };

  const handleSelectTicker = (ticker) => {
    console.log(`🎯 SELECTING TICKER: ${ticker}`);
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: 'SET_TICKER', ticker }));
    }
  };

  if (!globalData) return (
    <div className="h-screen w-full bg-black flex items-center justify-center text-emerald-500 font-mono">
      <Zap className="animate-pulse mr-2" /> INITIALIZING COCKPIT SYSTEM...
    </div>
  );

  return (
    <ErrorBoundary>
      <MissionManual isOpen={showManual} onClose={() => setShowManual(false)} />
      
      {/* Emergency Overlays */}
      {isKilled && (
        <div className="fixed top-0 left-0 w-full h-full z-[100] bg-red-950/20 pointer-events-none border-[10px] border-red-600/50 animate-pulse pointer-events-none" />
      )}

      <div className="min-h-screen w-full bg-black text-slate-300 font-mono text-xs flex flex-col p-2 gap-2">
      {/* Header */}
      <div className="flex-none h-16 flex justify-between items-center border-b border-slate-800 pb-2">
        <div className="flex items-center gap-4">
          <div className="text-lg font-bold text-emerald-500">UQTS-2026 MISSION CONTROL</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-red-500'}`} />
            <span className="uppercase text-[10px]">{status}</span>
          </div>
          <div className="text-[10px] text-slate-500 font-mono">KNOWLEDGE_TIME: {globalData.timestamp}</div>
          
          {alerts.length > 0 && (
             <div className="bg-red-600 text-white px-4 py-1 font-bold animate-bounce flex items-center gap-2 text-[10px] shadow-[0_0_15px_rgba(220,38,38,0.5)]">
                <ShieldAlert className="w-3 h-3" /> {alerts[alerts.length-1].msg}
             </div>
          )}
          
          <div className="flex items-center gap-6 px-6 border-l border-slate-800 ml-2">
             <div className="flex flex-col">
                <span className="text-[8px] text-slate-600 uppercase font-bold">Gross Exp</span>
                <span className="text-xs font-bold text-slate-100">{globalData.institutional?.gross_exposure.toFixed(1)}%</span>
             </div>
             <div className="flex flex-col">
                <span className="text-[8px] text-slate-600 uppercase font-bold">Net Exp</span>
                <span className={`text-xs font-bold ${Math.abs(globalData.institutional?.net_exposure) > 10 ? 'text-amber-500' : 'text-slate-100'}`}>
                  {globalData.institutional?.net_exposure > 0 ? '+' : ''}{globalData.institutional?.net_exposure.toFixed(1)}%
                </span>
             </div>
             <div className="flex flex-col border-l border-slate-800 pl-6">
                <span className="text-[8px] text-slate-600 uppercase font-bold">API Latency</span>
                <span className={`text-xs font-bold flex items-center gap-1.5 ${globalData.institutional?.data_latency_ms > 1000 ? 'text-amber-500' : 'text-emerald-500'}`}>
                  <Activity className="w-3 h-3" /> {globalData.institutional?.data_latency_ms.toFixed(0)}ms
                </span>
             </div>
             <div className="flex flex-col border-l border-slate-800 pl-6">
                <span className="text-[8px] text-slate-600 uppercase font-bold">Data Freshness</span>
                <span className={`text-xs font-bold flex items-center gap-1.5 ${globalData.institutional?.data_freshness_s > 65 ? 'text-red-500 animate-pulse' : 'text-slate-100'}`}>
                  <Zap className="w-3 h-3" /> {globalData.institutional?.data_freshness_s.toFixed(0)}s ago
                </span>
             </div>
          </div>

          <button 
            onClick={() => setShowManual(true)}
            className="flex items-center gap-1.5 px-3 py-1 border border-slate-700 hover:border-emerald-500/50 hover:bg-emerald-500/10 transition-all text-slate-400 hover:text-emerald-400 group"
          >
            <HelpCircle className="w-3 h-3" />
            <span className="text-[9px] uppercase font-bold tracking-widest">Mission Manual</span>
          </button>
        </div>
        <button 
          onClick={triggerKillSwitch}
          disabled={isKilled}
          className={`px-6 py-2 transition-all font-bold uppercase tracking-tighter border ${isKilled ? 'bg-red-600 text-white border-white cursor-not-allowed shadow-[0_0_20px_rgba(220,38,38,0.5)]' : 'bg-red-900/30 border-red-500 text-red-500 hover:bg-red-500 hover:text-white'}`}
        >
          {isKilled ? 'SYSTEM LIQUIDATED' : 'Emergency Kill Switch'}
        </button>
      </div>

      {/* Main Container - No longer fixed height */}
      <div className="flex-1 grid grid-cols-12 gap-2 pb-8">
        
        {/* 1. Spectral & Signal Viewer */}
        <Panel title="Spectral & Signal Viewer" icon={Activity} className="col-span-4 h-fit">
          <div className="flex flex-col gap-6">
            {spectralData && (
              <PriceChart data={spectralData.history} ticker={spectralData.ticker} />
            )}
            
            <div className="h-72 relative border border-slate-800 bg-black shadow-inner">
              {spectralData ? (
                <Heatmap data={spectralData.cwt} title={`${spectralData.ticker} Wavelet Spectrogram (Morlet)`} />
              ) : (
                <div className="h-full w-full flex items-center justify-center text-slate-600 italic">Select a ticker to load spectral data</div>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-800/50 p-6 border border-slate-700 flex flex-col justify-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-tighter leading-none mb-1">ADF p-value</div>
                <div className={`text-3xl font-bold ${spectralData && typeof spectralData.adf_p_value === 'number' && spectralData.adf_p_value < 0.05 ? 'text-emerald-500' : 'text-red-500 animate-pulse'}`}>
                  {spectralData && typeof spectralData.adf_p_value === 'number' ? spectralData.adf_p_value.toFixed(6) : "---"}
                </div>
                <div className="text-[8px] text-slate-600 mt-1 uppercase font-bold">Stationarity Test</div>
              </div>
              <div className="bg-slate-800/50 p-6 border border-slate-700 flex flex-col justify-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-tighter leading-none mb-1">Model State</div>
                <div className="flex gap-2 mt-1">
                  <span className="bg-emerald-900/40 text-emerald-400 px-2 py-1 border border-emerald-800 text-[10px] font-bold">STATIONARY</span>
                  <span className="bg-emerald-900/40 text-emerald-400 px-2 py-1 border border-emerald-800 text-[10px] font-bold">READY</span>
                </div>
              </div>
            </div>
            <div className="h-64 mt-2">
              <div className="text-[10px] text-slate-500 mb-4 uppercase tracking-widest text-right">Idiosyncratic Feature Importance (SHAP)</div>
              {spectralData && spectralData.shap_values ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Object.entries(spectralData.shap_values).map(([name, val]) => ({ name, val }))} layout="vertical">
                    <XAxis type="number" hide domain={[0, 1]} />
                    <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Bar dataKey="val" fill="#10b981" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full w-full flex items-center justify-center text-slate-800 border border-slate-900">NO SIGNAL</div>
              )}
            </div>
          </div>
        </Panel>

        {/* 2. Metacognition Panel */}
        <Panel title="Metacognition Panel" icon={ShieldAlert} className="col-span-4 h-fit">
          <div className="flex flex-col gap-8">
             <div className="flex justify-between items-center bg-slate-800/50 p-8 border border-slate-700 shadow-md">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Bayesian Belief Score</span>
                  <span className="text-5xl font-bold text-emerald-400">
                    {globalData && globalData.metacognition && typeof globalData.metacognition.belief_score === 'number' 
                      ? (globalData.metacognition.belief_score * 100).toFixed(2) 
                      : "---"}%
                  </span>
                </div>
                <Gauge className="w-16 h-16 text-emerald-500 opacity-40" />
             </div>
             
             <div className="h-80">
                <div className="text-[10px] text-slate-500 mb-4 uppercase tracking-widest text-center">Manifold Drift (t-SNE Latent Manifold)</div>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="x" name="Latent 1" stroke="#475569" fontSize={11} label={{ value: 'Latent 1', position: 'insideBottomRight', offset: -5, fill: '#475569' }} />
                    <YAxis type="number" dataKey="y" name="Latent 2" stroke="#475569" fontSize={11} label={{ value: 'Latent 2', angle: -90, position: 'insideLeft', fill: '#475569' }} />
                    <Scatter name="Training" data={globalData.metacognition.manifold_drift.slice(0, 5).map(p => ({ x: p[0], y: p[1] }))} fill="#475569" shape="circle" />
                    <Scatter name="Live" data={globalData.metacognition.manifold_drift.slice(5).map(p => ({ x: p[0], y: p[1] }))} fill="#10b981" shape="cross" />
                  </ScatterChart>
                </ResponsiveContainer>
             </div>

             <div className="h-64 mt-4">
                <div className="text-[10px] text-slate-500 mb-2 uppercase tracking-widest">Alpha Decay (Cumulative Information Gain)</div>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={globalData.metacognition.alpha_decay.map((val, i) => ({ i, val }))}>
                    <Line type="monotone" dataKey="val" stroke="#10b981" dot={false} strokeWidth={4} />
                    <XAxis hide />
                    <YAxis stroke="#475569" fontSize={10} label={{ value: 'Info Gain', angle: -90, position: 'insideLeft', fill: '#475569' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '11px' }} />
                  </LineChart>
                </ResponsiveContainer>
             </div>
          </div>
        </Panel>

        {/* 3. Cross-Sectional Ranking Grid */}
        <Panel title="Ranking Grid" icon={TrendingUp} className="col-span-4 h-fit">
          <div className="flex flex-col gap-4">
            <div className="h-[450px] flex flex-col">
               <RankingGrid ladder={globalData.rankings.ladder} onSelectTicker={handleSelectTicker} />
            </div>
            
            <div className="h-72 border-t border-slate-800 pt-6 mt-4 flex flex-col gap-4">
              <div className="text-[10px] text-slate-500 mb-2 uppercase font-bold tracking-widest">Sector Exposure (%)</div>
              <div className="flex-1 grid grid-cols-4 gap-2 overflow-y-auto no-scrollbar">
                {Object.entries(globalData.institutional?.sector_exposure || {}).map(([sector, exp]) => (
                   <div key={sector} className="bg-slate-800/30 border border-slate-800 p-2 flex flex-col justify-center">
                      <div className="text-[8px] text-slate-500 uppercase truncate">{sector}</div>
                      <div className={`text-xs font-bold ${exp > 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {exp > 0 ? '+' : ''}{exp.toFixed(1)}%
                      </div>
                   </div>
                ))}
              </div>
              
              <div className="text-[10px] text-slate-500 mb-2 mt-4 uppercase font-bold tracking-widest border-t border-slate-800 pt-4">L/S Equity Spread (Cumulative)</div>
              <ResponsiveContainer width="100%" height={120}>
                <LineChart 
                  data={globalData.rankings.ls_spread.map((val, i) => ({ i, val }))}
                  margin={{ left: 20, right: 10, top: 10, bottom: 20 }}
                >
                  <Line type="stepAfter" dataKey="val" stroke="#10b981" dot={false} strokeWidth={3} />
                  <CartesianGrid stroke="#1e293b" vertical={false} strokeDasharray="3 3" />
                  <XAxis hide />
                  <YAxis 
                    stroke="#475569" 
                    fontSize={10} 
                    label={{ value: 'Perf %', angle: -90, position: 'insideLeft', offset: -10, fill: '#475569' }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>

        {/* 4. Execution & Reality Check */}
        <Panel title="Execution & Reality Check" icon={Cpu} className="col-span-8 h-fit">
          <div className="grid grid-cols-12 h-full gap-8 p-4">
            <div className="col-span-3 flex flex-col justify-center gap-4 border-r border-slate-800 pr-8">
              <div className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Implementation Shortfall</div>
              <div className="text-4xl font-bold text-slate-100">
                {globalData && globalData.execution && typeof globalData.execution.implementation_shortfall === 'number'
                  ? globalData.execution.implementation_shortfall.toFixed(2)
                  : "---"}{" "}
                <span className="text-sm text-slate-600 font-normal">BPS</span>
              </div>
              <div className="text-[10px] text-slate-400 font-mono">VAR: {globalData.execution.is_var.toFixed(6)}</div>
              {globalData.execution.needs_retune && (
                <div className="text-[10px] text-red-500 flex items-center gap-2 animate-pulse bg-red-900/20 p-2 border border-red-900">
                  <AlertTriangle className="w-3 h-3" /> RL Agent: RETUNE REQUIRED
                </div>
              )}
            </div>

            <div className="col-span-4 border-r border-slate-800 pr-8">
              <div className="text-[10px] text-slate-500 mb-4 uppercase font-bold tracking-widest flex justify-between items-center">
                 <span>OMS Queue (Live)</span>
                 <div className="flex gap-2">
                    <span className="text-emerald-500">{globalData.institutional?.oms_queue.filled}F</span>
                    <span className="text-amber-500">{globalData.institutional?.oms_queue.working}W</span>
                    <span className="text-red-500">{globalData.institutional?.oms_queue.rejected}R</span>
                 </div>
              </div>
              <div className="flex flex-col gap-1.5 h-32 overflow-hidden bg-black/40 p-2 font-mono text-[9px]">
                 {globalData.institutional?.order_log.map((log, i) => (
                    <div key={i} className="flex justify-between border-l border-slate-800 pl-2">
                       <span className="text-slate-500">{log.time}</span>
                       <span className={log.side === 'BUY' ? 'text-emerald-500' : 'text-rose-500'}>{log.side} {log.ticker}</span>
                       <span className="text-slate-400">{log.qty}</span>
                       <span className={`font-bold ${log.status === 'REJECTED' ? 'text-red-500' : log.status === 'FILLED' ? 'text-emerald-500' : 'text-amber-500 animate-pulse'}`}>{log.status}</span>
                    </div>
                 ))}
              </div>
            </div>

            <div className="col-span-3">
              <div className="text-[10px] text-slate-500 mb-4 uppercase text-center font-bold tracking-widest">Slippage Heatmap</div>
              <div className="grid grid-cols-5 grid-rows-5 h-32 gap-[3px] bg-slate-800/20 p-2 border border-slate-800 shadow-inner">
                {globalData.execution.slippage_heatmap.flat().map((v, i) => (
                   <div key={i} className="bg-emerald-500 transition-all duration-300" style={{ opacity: v }} />
                ))}
              </div>
            </div>

            <div className="col-span-2 flex flex-col justify-center items-end text-right gap-2">
              <div className="text-[11px] text-slate-500 uppercase italic font-bold tracking-tighter">Execution Muscle</div>
              <div className="text-[10px] font-bold text-emerald-500 bg-emerald-900/20 px-2 py-1 border border-emerald-900 shadow-[0_0_10px_rgba(16,185,129,0.2)]">C++26 [LINKED]</div>
              <div className="text-[10px] text-slate-400 font-mono">LATENCY: 84.2μs</div>
              <div className="text-[9px] text-slate-600">STABILITY: NOMINAL</div>
            </div>
          </div>
        </Panel>

        {/* 5. Pipeline Control */}
        <Panel title="Research Pipeline Control" icon={Terminal} className="col-span-4 h-fit">
          <div className="flex flex-col gap-6 h-full justify-center">
            <div className="flex justify-around items-center border-b border-slate-800 pb-4">
               <div className="flex flex-col items-center">
                  <span className="text-slate-500 uppercase text-[10px] font-bold">Champion</span>
                  <span className="text-emerald-500 text-xl font-bold underline cursor-help">SHARPE {globalData.pipeline.champion_sharpe}</span>
               </div>
               <div className="flex flex-col items-center">
                  <span className="text-slate-500 uppercase text-[10px] font-bold">Challenger</span>
                  <span className="text-cyan-400 text-xl font-bold underline cursor-help">SHARPE {globalData.pipeline.challenger_sharpe}</span>
               </div>
            </div>
            <div className="overflow-hidden bg-black p-4 border border-slate-800 text-emerald-500/80 font-mono text-[11px] shadow-inner h-24">
               <span className="animate-pulse mr-2 text-white font-bold">&gt;</span> {globalData.pipeline.training_progress}
            </div>
          </div>
        </Panel>

      </div>
    </div>
    </ErrorBoundary>
  );
}
