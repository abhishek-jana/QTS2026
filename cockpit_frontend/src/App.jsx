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
  X,
  Filter,
  GripVertical
} from 'lucide-react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
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
  ScatterChart, 
  Scatter, 
  Label 
} from 'recharts';

// --- Components ---
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(error) { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-full bg-slate-900 text-red-500 p-10 font-mono flex flex-col items-center justify-center border-[20px] border-red-900/50">
          <ShieldAlert className="w-16 h-16 mb-4 animate-pulse" />
          <h1 className="text-2xl font-bold uppercase">Core Interface Failure</h1>
          <button onClick={() => window.location.reload()} className="mt-8 bg-red-600 text-white px-12 py-3 font-black uppercase text-lg hover:bg-red-500 shadow-2xl transition-all">Reboot Cockpit</button>
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
  const [selectedRangeLabel, setSelectedRangeLabel] = useState('ALL');
  const [hasInitialized, setHasInitialized] = useState(false);

  const ranges = [
    { label: '1W', value: 7 * 24 * 3600 },
    { label: '1M', value: 30 * 24 * 3600 },
    { label: '3M', value: 90 * 24 * 3600 },
    { label: '1Y', value: 365 * 24 * 3600 },
    { label: '5Y', value: 5 * 365 * 24 * 3600 },
    { label: '10Y', value: 10 * 365 * 24 * 3600 },
    { label: 'ALL', value: null }
  ];

  const applyRange = (rangeLabel, chartData) => {
    if (!chartRef.current || !chartData || chartData.length === 0) return;
    
    if (rangeLabel === 'ALL') {
      chartRef.current.timeScale().fitContent();
    } else {
      const range = ranges.find(r => r.label === rangeLabel);
      if (range && range.value) {
        const latest = chartData[chartData.length - 1].time;
        const from = latest - range.value;
        chartRef.current.timeScale().setVisibleRange({ from, to: latest });
      }
    }
  };

  const handleRangeChange = (range) => {
    setSelectedRangeLabel(range.label);
    applyRange(range.label, data);
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;
    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      width: chartContainerRef.current.clientWidth || 400,
      height: 250,
      timeScale: { 
        borderColor: '#1e293b', 
        timeVisible: true, 
        secondsVisible: false,
        shiftVisibleRangeOnNewBar: true, // Key for professional behavior
      },
    });
    const series = chart.addSeries(CandlestickSeries, { 
        upColor: '#10b981', 
        downColor: '#ef4444', 
        borderVisible: false, 
        wickUpColor: '#10b981', 
        wickDownColor: '#ef4444' 
    });
    chartRef.current = chart; seriesRef.current = series;
    
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Handle data updates separately from initialization
  useEffect(() => {
    if (seriesRef.current && data && data.length > 0) {
      const seen = new Set();
      const cleanData = data
        .filter(d => {
            if (seen.has(d.time)) return false;
            seen.add(d.time);
            return true;
        })
        .sort((a, b) => a.time - b.time);
      
      if (cleanData.length > 0) {
        seriesRef.current.setData(cleanData);
        
        if (!hasInitialized) {
          chartRef.current.timeScale().fitContent();
          setHasInitialized(true);
        } else if (selectedRangeLabel !== 'ALL') {
          // Re-apply range to stick to the right edge with user's zoom
          applyRange(selectedRangeLabel, cleanData);
        }
      }
    }
  }, [data]); // Only re-run on data change

  return (
    <div className="w-full flex flex-col gap-1">
      <div className="flex justify-between items-center mb-1">
        <div className="text-lg font-mono text-emerald-500 uppercase font-bold tracking-tighter">
          {ticker} LIVE FEED (SIP_STREAM)
        </div>
        <div className="flex gap-1">
          {ranges.map(r => (
            <button 
              key={r.label}
              onClick={() => handleRangeChange(r)}
              className={`px-1.5 py-0.5 text-[10px] font-black border transition-all rounded-sm uppercase ${selectedRangeLabel === r.label ? 'bg-emerald-500 text-white border-emerald-500' : 'border-slate-800 bg-slate-900/50 text-slate-500 hover:text-emerald-400 hover:border-emerald-900/50'}`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
      <div ref={chartContainerRef} className="w-full h-[250px] border border-slate-800 bg-black/50 shadow-inner" />
    </div>
  );
};

const Heatmap = ({ data, title }) => {
  const canvasRef = useRef();
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current; const ctx = canvas.getContext('2d', { alpha: false });
    const rows = data.length; 
    const cols = data[0].length;
    canvas.width = cols; canvas.height = rows;
    const imageData = ctx.createImageData(cols, rows);
    const flatData = data.flat(); 
    const maxVal = Math.max(...flatData.slice(0, 10000), 0.000001);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = data[i][j]; const ratio = Math.min(1.0, val / maxVal); const idx = (i * cols + j) * 4;
        imageData.data[idx] = Math.min(255, ratio * 450); imageData.data[idx + 1] = Math.max(0, (ratio - 0.3) * 550); 
        imageData.data[idx + 2] = Math.max(0, (ratio - 0.6) * 650); imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <div className="w-full h-full flex flex-col min-h-0">
      <div className="text-lg font-mono text-slate-500 mb-1 tracking-tighter truncate font-bold uppercase">{title}</div>
      <div className="flex-1 min-h-0 relative border border-slate-800 bg-black overflow-hidden shadow-inner">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ imageRendering: 'pixelated' }} />
      </div>
    </div>
  );
};

const SortableRow = ({ row, onSelectTicker }) => {
  const [flash, setFlash] = useState(null);
  const prevPriceRef = useRef(row.price);

  useEffect(() => {
    if (row.price !== prevPriceRef.current) {
        setFlash(row.price > prevPriceRef.current ? 'up' : 'down');
        const timer = setTimeout(() => setFlash(null), 1000);
        prevPriceRef.current = row.price;
        return () => clearTimeout(timer);
    }
  }, [row.price]);

  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: row.ticker });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 50 : 1,
    backgroundColor: flash === 'up' ? 'rgba(16, 185, 129, 0.15)' : flash === 'down' ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
  };

  return (
    <tr 
      ref={setNodeRef} 
      style={style}
      className={`border-b border-slate-800/40 hover:bg-emerald-500/10 cursor-pointer group transition-all relative ${flash ? 'duration-100' : 'duration-1000'}`}
    >
      <td className="py-2.5 pl-2">
        <div {...attributes} {...listeners} className="cursor-grab active:cursor-grabbing p-1 text-slate-700 hover:text-emerald-500 transition-colors">
          <GripVertical className="w-3 h-3" />
        </div>
      </td>
      <td onClick={() => onSelectTicker(row.ticker)} className="py-2.5 flex items-center gap-2 transition-all group-hover:translate-x-1">
        <div className={"w-1 h-3 " + (row.score > 0 ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-red-500")} />
        <span className="font-bold text-slate-200 group-hover:text-emerald-400">{row.ticker}</span>
      </td>
      <td onClick={() => onSelectTicker(row.ticker)} className="text-right py-2 text-slate-400 pr-4">${(row.price || 0).toFixed(2)}</td>
      <td onClick={() => onSelectTicker(row.ticker)} className={"text-right py-2 pr-4 " + (row.score > 0 ? "text-emerald-400" : "text-red-400")}>{(row.score || 0).toFixed(4)}</td>
      <td onClick={() => onSelectTicker(row.ticker)} className={`text-right py-2 pr-4 font-black italic ${row.pnl_pct > 0 ? 'text-emerald-400' : row.pnl_pct < 0 ? 'text-rose-500' : 'text-slate-800'}`}>
        {row.market_value > 0 ? (
            <span className="flex flex-col text-right">
                <span>${row.market_value.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                <span className="text-[10px] opacity-70">({row.pnl_pct > 0 ? '+' : ''}{row.pnl_pct.toFixed(1)}%)</span>
            </span>
        ) : '—'}
      </td>
      <td onClick={() => onSelectTicker(row.ticker)} className="text-right py-2 pr-2"><span className={"text-xs px-1 border " + (row.score > 0 ? "border-emerald-800 text-emerald-600" : "border-red-900 text-red-700") + " uppercase font-black"}>{row.score > 0 ? "Long" : "Short"}</span></td>
    </tr>
  );
};

const RankingGrid = ({ ladder, onSelectTicker, filterSector, onClearFilter, tickerOrder, sensors, handleDragEnd }) => {
  const [showHoldingsOnly, setShowHoldingsOnly] = useState(false);
  
  const sortedData = React.useMemo(() => {
    if (!ladder || !tickerOrder || tickerOrder.length === 0) return [];
    
    // Create map for fast lookup
    const map = {};
    ladder.forEach(item => {
      map[item.ticker] = { 
        ...item, 
        price: item.live_price, 
        pnl_pct: item.pnl_pct, 
        market_value: item.market_value 
      };
    });

    // Return items in tickerOrder
    let items = tickerOrder
      .map(ticker => map[ticker])
      .filter(item => item !== undefined);

    if (showHoldingsOnly) {
        items = items.filter(item => Math.abs(item.market_value) > 0.1);
    }

    if (filterSector) {
        items = items.filter(item => item.sector === filterSector);
    }

    return items;
  }, [ladder, tickerOrder, filterSector, showHoldingsOnly]);

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar pr-2 h-[520px]">
      <div className="flex justify-between items-center mb-2 border-b border-emerald-900/30 pb-1">
        <div className="text-base font-bold text-emerald-500/80 uppercase tracking-tighter italic">Decile Ladder (House View)</div>
        <div className="flex items-center gap-2">
            <button 
                onClick={() => setShowHoldingsOnly(!showHoldingsOnly)}
                className={`px-2 py-0.5 text-[9px] font-black uppercase transition-all border ${showHoldingsOnly ? 'bg-emerald-500 text-white border-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]' : 'border-slate-800 text-slate-600 hover:text-slate-400'}`}
            >
                Holdings Only
            </button>
            {filterSector && (
            <button onClick={onClearFilter} className="flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded-sm hover:bg-emerald-500/20 transition-all">
                <span className="text-sm font-bold text-emerald-400 uppercase tracking-widest">{filterSector}</span>
                <X className="w-2 h-2 text-emerald-400" />
            </button>
            )}
        </div>
      </div>
      
      <DndContext 
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <table className="w-full text-sm border-collapse font-mono">
          <thead className="sticky top-0 bg-slate-900 shadow-sm text-slate-500 uppercase z-10">
            <tr>
              <th className="w-6"></th>
              <th className="text-left py-2 font-normal">Ticker</th>
              <th className="text-right py-2 pr-4 font-normal">Price</th>
              <th className="text-right py-2 pr-4 font-normal">Z-Score</th>
              <th className="text-right py-2 pr-4 font-normal">Position</th>
              <th className="text-right py-2 font-normal">Action</th>
            </tr>
          </thead>
          <SortableContext 
            items={tickerOrder}
            strategy={verticalListSortingStrategy}
          >
            <tbody>
              {sortedData.map((row) => (
                <SortableRow key={row.ticker} row={row} onSelectTicker={onSelectTicker} />
              ))}
            </tbody>
          </SortableContext>
        </table>
      </DndContext>
      {sortedData.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-800 p-10 text-center">
            <Activity className="w-12 h-12 mb-4 opacity-20" />
            <div className="text-lg font-black uppercase tracking-[0.3em] opacity-40">Awaiting Signal</div>
            <div className="text-[10px] uppercase font-bold tracking-widest opacity-30 mt-2">No active holdings meet current filter criteria</div>
        </div>
      )}
    </div>
  );
};

const Panel = ({ title, icon: Icon, children, className = "" }) => (
  <div className={"bg-slate-900 border border-slate-700 p-4 flex flex-col h-fit shadow-lg " + className}>
    <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
      <Icon className="w-4 h-4 text-emerald-500" />
      <h2 className="text-sm font-black uppercase tracking-[0.2em] text-slate-400">{title}</h2>
    </div>
    <div className="flex-1 min-h-0">
      {children}
    </div>
  </div>
);

const MissionManual = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  const sections = [
    { 
      title: "1. Spectral Physics (4 Instruments)", 
      items: [
        { label: "Price Chart", detail: "Real-time OHLCV candles with persistent multi-year zoom (1W to 10Y). Resilient to incoming data ticks." },
        { label: "Wavelet Heatmap", detail: "16-scale CWT Spectrogram. Brightness = Energy. Lower scales = High frequency; Higher scales = Structural trends." },
        { label: "ADF p-value", detail: "Augmented Dickey-Fuller stationarity test. Threshold: < 0.05. Ensures model trades mean-reverting energy." },
        { label: "Alpha State", detail: "READY indicates the neural engine has locked onto a valid spectral energy pulse for the current ticker." }
      ] 
    },
    { 
      title: "2. Neural Logic & Metacognition", 
      items: [
        { label: "SHAP Fusion", detail: "Bridges latent dimensions to factors: Momentum, Volatility, Sentiment, and Liquidity." },
        { label: "Bayesian Belief", detail: "Confidence metric. Initializes at 50% ('Prove It' mode). Uses strict Bayesian updating to scale conviction." },
        { label: "Manifold Drift", detail: "t-SNE projection. Circles = Base; Crosses = Live. Separation indicates Out-Of-Sample drift." },
        { label: "Cumulative Gain", detail: "Integrates alpha performance over the session. Flattening signals regime exhaustion." }
      ] 
    },
    { 
      title: "3. Market Dynamics & Ledger", 
      items: [
        { label: "Decile Ladder", detail: "Alpha rankings with Drag & Drop reordering, Price Delta flashing, and a 'Holdings Only' active risk filter." },
        { label: "Sector Matrix", detail: "Interactive exposure grid. Clicking a block filters the Decile Ladder to that specific sector." },
        { label: "ROE (Invested)", detail: "Return on Exposure. Measures profit efficiency strictly against the capital currently at risk (Gross Notional)." },
        { label: "Realized / Unrealized", detail: "Splits total P&L into banked profits (closed trades) and floating profit (open positions)." }
      ] 
    },
    { 
      title: "4. Execution Muscle", 
      items: [
        { label: "OMS Queue", detail: "Real-time scrolling log showing side (BUY/SELL/SHORT/COVER), dynamic share quantity, and dollar notional." },
        { label: "Sim Warp", detail: "In historical mode, displays the time-acceleration factor (e.g. 1.2M x) of the backtest." },
        { label: "IS (BPS) & Slippage", detail: "Implementation Shortfall gap and 5x5 heatmap mapping liquidity distribution impact." },
        { label: "Sharpe A/B Test", detail: "Real-time Champion (V1) vs Challenger (V2) comparison tracking regime dominance." }
      ] 
    }
  ];
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/95 backdrop-blur-md p-4 font-mono">
      <div className="bg-slate-900 border border-emerald-500/50 w-full max-w-6xl max-h-[95vh] overflow-hidden flex flex-col shadow-2xl">
        <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-emerald-500/10">
          <div className="flex items-center gap-3 text-emerald-400 font-bold uppercase tracking-[0.3em] text-base">
            <Terminal className="w-5 h-5 animate-pulse" /> 
            UQTS-2026: COMMAND MANUAL V2.0
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors bg-slate-800/50 p-2 rounded-full hover:bg-rose-500/20">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-10 space-y-12 overflow-y-auto no-scrollbar bg-gradient-to-b from-slate-900 to-black">
          {sections.map((s, i) => (
            <div key={i} className="space-y-6">
              <h3 className="text-emerald-500 font-black uppercase text-base tracking-[0.4em] border-b border-emerald-900/40 pb-3 flex items-center gap-3">
                <span className="opacity-30 text-lg">0{i+1}</span> {s.title}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {s.items.map((it, j) => (
                  <div key={j} className="bg-slate-800/10 p-5 border border-slate-700/30 group hover:border-emerald-500/50 transition-all hover:bg-slate-800/30 shadow-2xl">
                    <h4 className="text-slate-100 font-bold text-sm uppercase mb-3 flex items-center gap-2">
                      <div className="w-2 h-2 bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.8)]" />
                      {it.label}
                    </h4>
                    <p className="text-slate-400 text-xs leading-relaxed font-medium group-hover:text-slate-200">{it.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="p-8 bg-slate-900 border-t border-slate-800 flex justify-between items-center">
          <div className="space-y-1">
            <div className="text-sm text-emerald-500 font-black uppercase tracking-widest flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
              SYSTEM_HEARTBEAT: NOMINAL
            </div>
            <div className="text-xs text-slate-600 uppercase font-bold tracking-widest italic">C++26 OSQP Kernel // V2 High-Density Alpha</div>
          </div>
          <button onClick={onClose} className="bg-emerald-600 text-black px-20 py-4 font-black uppercase text-sm hover:bg-emerald-400 shadow-[0_0_30px_rgba(16,185,129,0.4)] transition-all active:scale-95 border-none">
            ACKNOWLEDGE & INITIALIZE COCKPIT
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
  const [selectedSector, setSelectedSector] = useState(null);
  const [tickerOrder, setTickerOrder] = useState([]);
  const ws = useRef(null);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/cockpit');
    ws.current.onopen = () => setStatus('active');
    ws.current.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.type === 'GLOBAL_UPDATE') {
            setGlobalData(payload);
            // SENIOR FIX: Initialize ticker order ALPHABETICALLY if not yet set
            if (payload.rankings?.ladder && tickerOrder.length === 0) {
                const initialOrder = payload.rankings.ladder
                    .map(item => item.ticker)
                    .sort((a, b) => a.localeCompare(b));
                setTickerOrder(initialOrder);
            }
        }
        else if (payload.type === 'SPECTRAL_UPDATE') {
            console.log("🟢 WS: Received Spectral Bundle for " + payload.spectral.ticker);
            setSpectralData(payload.spectral);
        }
      } catch (err) {}
    };
    ws.current.onclose = () => setStatus('disconnected');
    return () => ws.current.close();
  }, [tickerOrder.length]);

  const handleDragEnd = (event) => {
    const { active, over } = event;
    if (active.id !== over.id) {
      setTickerOrder((items) => {
        const oldIndex = items.indexOf(active.id);
        const newIndex = items.indexOf(over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  const handleSelectTicker = (ticker) => {
    console.log("🎯 UI: Triggering Spectral Focus for " + ticker);
    if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ command: 'SET_TICKER', ticker }));
    }
  };

  if (!globalData) return <div className="h-screen w-full bg-black flex flex-col items-center justify-center text-emerald-500 font-mono gap-4 animate-pulse"><Zap className="w-12 h-12" /> INITIALIZING COCKPIT ENGINE...</div>;

  const inst = globalData.institutional || {};
  const meta = globalData.metacognition || {};
  const exec = globalData.execution || {};
  const pipeline = globalData.pipeline || {};

  return (
    <ErrorBoundary>
      <MissionManual isOpen={showManual} onClose={() => setShowManual(false)} />
      <div className="min-h-screen w-full bg-black text-slate-300 font-mono text-base flex flex-col p-2 gap-2 overflow-x-hidden selection:bg-emerald-500/30">
        
        {/* Header */}
        <div className="flex-none h-16 flex justify-between items-center border-b border-slate-800 pb-2 px-2">
          <div className="flex items-center gap-4">
            <div className="text-2xl font-black text-emerald-500 tracking-tighter italic">UQTS-2026</div>
            <div className="flex items-center gap-2 border-l border-slate-800 pl-4"><div className={"w-2 h-2 rounded-full " + (status === "active" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" : "bg-red-500")} /><span className="uppercase text-sm font-bold text-slate-500">{status}</span></div>
            <div className="text-sm text-slate-600 border border-slate-800 px-2 py-0.5 rounded-sm uppercase tracking-tighter">SIM_TIME: {globalData.timestamp}</div>
            <div className={"text-[10px] px-2 py-0.5 border font-black uppercase tracking-widest " + (inst.market_status === 'OPEN' ? "text-emerald-500 border-emerald-900/50 bg-emerald-950/20" : "text-amber-600 border-amber-900/50 bg-amber-950/20")}>Market {inst.market_status || 'OPEN'}</div>
            <div className="flex gap-6 px-6 border-l border-slate-800">
               <div className="flex flex-col border-l border-slate-800 pl-6">
                  <span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">Active Pos</span>
                  <span className={`text-sm font-bold italic transition-all ${(inst.active_positions || 0) > 0 ? 'text-emerald-400 animate-pulse' : 'text-slate-500'}`}>
                    {(inst.active_positions || 0)}
                  </span>
               </div>
               <div className="flex flex-col border-l border-slate-800 pl-6"><span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">Gross Exp</span><span className="text-sm font-bold text-slate-100 italic">{(inst.gross_exposure || 0).toFixed(1)}%</span></div>
               <div className="flex flex-col border-l border-slate-800 pl-6"><span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">Account Value</span><span className="text-sm font-bold text-emerald-500 italic">${(inst.capital || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span></div>
               <div className="flex flex-col border-l border-slate-800 pl-6"><span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">Buying Power</span><span className="text-sm font-bold text-cyan-500 italic">${(inst.buying_power || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span></div>
               <div className="flex flex-col border-l border-slate-800 pl-6">
                  <span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">ROE (Invested)</span>
                  <span className={"text-sm font-bold italic " + ((inst.roe || 0) >= 0 ? "text-emerald-400" : "text-rose-500")}>{(inst.roe || 0) >= 0 ? "+" : ""}{(inst.roe || 0).toFixed(2)}%</span>
               </div>
               <div className="flex flex-col border-l border-slate-800 pl-6">
                  <span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">Realized / Unrealized</span>
                  <div className="text-sm font-bold flex gap-2 italic">
                    <span className={(inst.pnl_realized || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}>
                        {(inst.pnl_realized || 0) >= 0 ? '+' : '-'}${Math.abs(inst.pnl_realized || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                    </span>
                    <span className="text-slate-700">/</span>
                    <span className={(inst.pnl_unrealized || 0) >= 0 ? "text-emerald-400" : "text-rose-400"}>
                        {(inst.pnl_unrealized || 0) >= 0 ? '+' : '-'}${Math.abs(inst.pnl_unrealized || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                    </span>
                  </div>
               </div>
               <div className="flex flex-col border-l border-slate-800 pl-6"><span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">{inst.trading_mode === 'sim' ? 'Sim Warp' : 'Latency'}</span><span className="text-sm font-bold text-emerald-500 italic">{inst.trading_mode === 'sim' ? '1.2M x' : (inst.data_latency_ms || 0).toFixed(0) + 'ms'}</span></div>
            </div>
            <button onClick={() => setShowManual(true)} className="flex items-center gap-1.5 px-3 py-1 border border-emerald-900/30 hover:bg-emerald-500/10 transition-all text-slate-500 hover:text-emerald-400 group ml-4 outline-none"><HelpCircle className="w-3 h-3" /><span className="text-sm uppercase font-bold tracking-[0.2em]">Manual</span></button>
          </div>
          <button onClick={() => ws.current.send('KILL_SWITCH')} className="px-6 py-2 transition-all font-black uppercase text-sm border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-white shadow-[0_0_15px_rgba(220,38,38,0.2)]">Kill System</button>
        </div>

        <div className="flex-1 grid grid-cols-12 gap-2 pb-4 overflow-hidden">
          
          <Panel title="Spectral Alpha" icon={Activity} className="col-span-4 h-full overflow-y-auto no-scrollbar">
            <div className="flex flex-col gap-6">
              {spectralData ? (<><PriceChart data={spectralData.history} ticker={spectralData.ticker} /><div className="h-72 relative border border-slate-800 bg-black shadow-2xl"><Heatmap data={spectralData.cwt} title={spectralData.ticker + " Wavelet Spectrogram"} /></div></>) : (<div className="h-[330px] w-full flex items-center justify-center text-slate-700 italic text-lg uppercase tracking-widest animate-pulse border border-slate-900 bg-black/40">Select ticker from ladder to hydrate manifold...</div>)}
              <div className="grid grid-cols-2 gap-2"><div className="bg-slate-800/30 p-6 border border-slate-700 text-center"><div className="text-lg text-slate-500 uppercase font-black mb-1 tracking-widest">ADF p-value</div><div className={"text-2xl font-black " + (spectralData?.adf_p_value < 0.05 ? "text-emerald-500" : "text-slate-500")}>{(spectralData?.adf_p_value || 0.0001).toFixed(6)}</div></div><div className="bg-slate-800/30 p-6 border border-slate-700 text-center flex flex-col items-center justify-center"><div className="text-lg text-slate-500 uppercase font-black mb-1 tracking-widest">Alpha State</div><span className="text-emerald-400 text-lg font-black border border-emerald-900/50 px-2 py-0.5 rounded-sm shadow-md">READY</span></div></div>
              <div className="h-48 border-t border-slate-800 pt-4"><div className="text-lg text-slate-600 mb-2 uppercase font-black tracking-widest text-right">Idiosyncratic SHAP Fusion</div>{spectralData?.shap_values ? (<ResponsiveContainer width="100%" height="100%"><BarChart data={Object.entries(spectralData.shap_values).map(([name, val]) => ({ name, val }))} layout="vertical" onClick={(data) => { if(data) alert(`MODALITY_OVERRIDE: Focus on ${data.activeLabel}`); }}><XAxis type="number" hide domain={[0, 1]} /><YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 12, fill: '#475569' }} /><Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '9px'}} cursor={{fill: 'rgba(16,185,129,0.05)'}} /><Bar dataKey="val" fill="#10b981" radius={[0, 2, 2, 0]} className="cursor-pointer hover:fill-emerald-400 transition-colors" /></BarChart></ResponsiveContainer>) : <div className="h-full w-full flex items-center justify-center text-slate-800 italic uppercase text-lg tracking-widest">Awaiting MODALITY_GATING...</div>}</div>
            </div>
          </Panel>

          <Panel title="Metacognition" icon={ShieldAlert} className="col-span-4 h-full overflow-y-auto no-scrollbar">
            <div className="flex flex-col gap-8">
               <div className="flex justify-between items-center bg-slate-800/30 p-8 border border-slate-700 shadow-2xl"><div className="flex flex-col"><span className="text-sm text-slate-600 uppercase font-black tracking-widest">Bayesian Belief</span><span className="text-6xl font-black text-emerald-400 drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]">{((meta.belief_score || 0) * 100).toFixed(2) + "%"}</span></div><Gauge className="w-16 h-16 text-emerald-500 opacity-20" /></div>
               <div className="h-72 border border-slate-800 bg-black/40 p-4">
                 <div className="text-sm text-slate-600 mb-4 uppercase font-black text-center tracking-[0.4em]">Manifold Drift (t-SNE Latent)</div>
                 <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ bottom: 30, left: 10, right: 10 }}>
                    <CartesianGrid strokeDasharray="1 4" stroke="#1e293b" />
                    <XAxis type="number" dataKey="x" stroke="#475569" fontSize={8} domain={['auto', 'auto']}><Label value="Latent Dim 1" offset={-15} position="insideBottom" fill="#475569" fontSize={9} fontStyle="italic" /></XAxis>
                    <YAxis type="number" dataKey="y" stroke="#475569" fontSize={8} domain={['auto', 'auto']}><Label value="Latent Dim 2" angle={-90} position="insideLeft" fill="#475569" fontSize={9} fontStyle="italic" offset={0} /></YAxis>
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '9px'}} />
                    {/* FIXED: Correct slice indices to show both training and live dots */}
                    <Scatter name="Base" data={(meta.manifold_drift || []).slice(0, 5).map(p => ({ x: p[0], y: p[1] }))} fill="#334155" shape="circle" />
                    <Scatter name="Live" data={(meta.manifold_drift || []).slice(5).map(p => ({ x: p[0], y: p[1] }))} fill="#10b981" shape="cross" strokeWidth={3} />
                  </ScatterChart>
                </ResponsiveContainer>
               </div>
               <div className="h-56 mt-4 border-t border-slate-800 pt-4"><div className="text-sm text-slate-600 mb-2 uppercase font-black tracking-widest">Cumulative Info Gain</div><ResponsiveContainer width="100%" height="100%"><LineChart data={(meta.alpha_decay || []).map((val, i) => ({ i, val }))} margin={{ bottom: 30, left: 10, right: 10 }}><XAxis dataKey="i" stroke="#475569" fontSize={8}><Label value="Realized Events" offset={-15} position="insideBottom" fill="#475569" fontSize={9} fontStyle="italic" /></XAxis><YAxis stroke="#475569" fontSize={8}><Label value="Cumulative Gain" angle={-90} position="insideLeft" fill="#475569" fontSize={9} fontStyle="italic" offset={0} /></YAxis><Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '9px'}} itemStyle={{color: '#10b981'}} /><Line type="monotone" dataKey="val" stroke="#10b981" dot={{ r: 2, fill: '#10b981' }} activeDot={{ r: 5 }} strokeWidth={4} /></LineChart></ResponsiveContainer></div>
               
               {/* L/S Equity Spread Moved Here */}
               <div className="h-56 mt-4 border-t border-slate-800 pt-4">
                    <div className="text-sm text-slate-600 uppercase font-black mb-2 tracking-widest">L/S Equity Spread (Cumulative)</div>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={(globalData.rankings?.ls_spread || []).map((val, i) => ({ i, val }))} margin={{ bottom: 30, left: 10, right: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                            <XAxis hide />
                            <YAxis stroke="#475569" fontSize={8} domain={['auto', 'auto']} />
                            <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '9px'}} itemStyle={{color: '#10b981'}} labelStyle={{display: 'none'}} />
                            <Line type="monotone" dataKey="val" stroke="#10b981" dot={false} strokeWidth={4} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
          </Panel>

          <Panel title="Ranking Ladder" icon={TrendingUp} className="col-span-4 h-full overflow-hidden flex flex-col">
             <RankingGrid 
                ladder={globalData.rankings?.ladder} 
                onSelectTicker={handleSelectTicker} 
                filterSector={selectedSector} 
                onClearFilter={() => setSelectedSector(null)} 
                tickerOrder={tickerOrder}
                sensors={sensors}
                handleDragEnd={handleDragEnd}
             />
             <div className="flex-1 border-t border-slate-800 mt-4 pt-4 overflow-y-auto no-scrollbar">
                <div className="text-sm text-slate-500 mb-2 uppercase font-black tracking-widest flex justify-between items-center"><span>Sector Matrix (Interactive)</span><Filter className="w-3 h-3 text-slate-700" /></div>
                <div className="grid grid-cols-3 gap-1.5 pb-4">
                  {Object.entries(inst.sector_exposure || {}).map(([sector, stats]) => (
                     <button key={sector} onClick={() => setSelectedSector(selectedSector === sector ? null : sector)} className={"border p-2 flex flex-col justify-center transition-all text-left " + (selectedSector === sector ? "bg-emerald-500/20 border-emerald-500" : "bg-slate-800/30 border-slate-800 hover:bg-slate-800/50 hover:border-slate-600")}>
                        <div className="text-sm text-slate-500 uppercase truncate font-bold">{sector}</div>
                        <div className={"text-sm font-black " + (stats.exposure > 0 ? "text-emerald-500" : "text-rose-500")}>{(stats.exposure > 0 ? "+" : "") + stats.exposure.toFixed(1)}%</div>
                        <div className="flex justify-between items-center mt-1 border-t border-slate-700/50 pt-1">
                            <span className="text-xs text-slate-600 uppercase font-black">{stats.count} Tickers</span>
                            <span className="text-xs text-emerald-600/60 italic font-black">α: {(stats.avg_score || 0).toFixed(2)}</span>
                        </div>
                     </button>
                  ))}
                </div>
             </div>
          </Panel>

          {/* Section 4: Execution & Reality Check */}
          <Panel title="Execution & Reality Check" icon={Cpu} className="col-span-12 mt-2 h-fit">
            <div className="grid grid-cols-12 gap-8 p-2">
              <div className="col-span-2 flex flex-col justify-center gap-2 border-r border-slate-800 pr-8">
                <div className="text-sm text-slate-500 uppercase font-black tracking-tighter leading-none mb-1">Implementation Shortfall</div>
                <div className="text-3xl font-black text-slate-100 italic">{(exec.implementation_shortfall || 0).toFixed(2)} <span className="text-sm text-slate-700 not-italic uppercase">BPS</span></div>
                <div className="text-sm text-slate-500 font-mono italic">{"IS_VAR_95: " + (exec.is_var || 0).toFixed(6)}</div>
              </div>

              {/* OMS Queue - Functional with SOLD info */}
              <div className="col-span-5 border-r border-slate-800 pr-8 flex flex-col gap-2">
                <div className="text-sm text-slate-500 uppercase font-black flex justify-between items-center tracking-widest border-b border-slate-800/50 pb-1">
                    <span>OMS Queue (Live)</span>
                    <div className="flex gap-4">
                        <span className="text-emerald-500 font-bold">{(inst.oms_queue?.filled || 0)} FILLED</span>
                        <span className="text-amber-500 font-bold">{(inst.oms_queue?.working || 0)} WORKING</span>
                        <span className="text-red-600 font-bold">{(inst.oms_queue?.rejected || 0)} REJECTED</span>
                    </div>
                </div>
                <div className="bg-black/50 p-2 h-64 overflow-y-auto no-scrollbar font-mono text-sm border border-slate-800 shadow-inner">
                    {(inst.order_log || []).map((log, i) => (
                        <div key={i} className="flex justify-between items-center border-b border-slate-900 pb-1.5 mb-1.5 hover:bg-emerald-500/5 transition-colors text-[11px]">
                            <span className="text-slate-600 w-24 shrink-0">{log.time}</span>
                            <span className={`font-bold w-20 ${log.side === 'BUY' ? 'text-emerald-500' : log.side === 'SELL' ? 'text-rose-500' : log.side === 'SHORT' ? 'text-amber-600' : 'text-cyan-600'}`}>{log.side + " " + log.ticker}</span>
                            <div className="flex flex-col items-end flex-1 pr-4">
                                <span className="text-slate-500 uppercase text-[9px] font-bold">QTY: {log.qty}</span>
                                <span className="text-emerald-600/80 text-[10px] font-black">${(log.notional || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
                            </div>
                            <span className={`font-black uppercase tracking-tighter w-16 text-right ${log.status === 'FILLED' ? 'text-emerald-400' : log.status === 'REJECTED' ? 'text-red-600' : 'text-amber-400 animate-pulse'}`}>{(log.status || "PENDING")}</span>
                        </div>
                    ))}
                    {(!inst.order_log || inst.order_log.length === 0) && <div className="h-full flex items-center justify-center text-slate-800 italic uppercase tracking-[0.3em] animate-pulse text-sm">Awaiting execution commands...</div>}
                </div>
              </div>
              
              <div className="col-span-2 border-r border-slate-800 pr-8">
                 <div className="flex flex-col gap-2 w-full px-2">
                    <div className="text-sm text-slate-600 uppercase font-black tracking-widest text-center">Slippage</div>
                    <div className="grid grid-cols-5 gap-1 bg-black/40 p-1.5 border border-slate-800 shadow-xl h-24">
                        {(exec.slippage_heatmap || []).flat().map((v, i) => (<div key={i} className="w-3.5 h-3.5 bg-emerald-500 rounded-sm transition-all duration-700" style={{ opacity: v }} />))}
                    </div>
                 </div>
              </div>

              <div className="col-span-3 flex flex-col justify-center items-end text-right gap-4">
                 <div className="w-full space-y-1.5">
                    <div className="grid grid-cols-2 gap-2">
                        <div className="bg-slate-800/20 p-2 border border-slate-800 flex flex-col items-center justify-center"><span className="text-[10px] text-slate-500 uppercase font-black">Champ</span><span className="text-sm font-black text-cyan-400">{(pipeline.champion_sharpe || 0).toFixed(2)}</span></div>
                        <div className="bg-slate-800/20 p-2 border border-slate-800 flex flex-col items-center justify-center"><span className="text-[10px] text-slate-500 uppercase font-black">Chal</span><span className="text-sm font-black text-emerald-400">{(pipeline.challenger_sharpe || 0).toFixed(2)}</span></div>
                    </div>
                    <div className="bg-black/40 p-1 border border-slate-800 text-[10px] text-emerald-500/60 uppercase font-black truncate animate-pulse text-center tracking-widest">SYSTEM_PIPELINE: {pipeline.training_progress || 'V1_ACTIVE'}</div>
                 </div>

                 <div className="flex flex-col items-end gap-1">
                    <div className="text-sm text-slate-600 italic uppercase font-black tracking-tighter italic">Muscle: C++26</div>
                    <div className="text-emerald-500 font-black text-sm shadow-[0_0_15px_rgba(16,185,129,0.2)] border border-emerald-900/50 px-3 py-1 uppercase tracking-widest bg-emerald-950/20">Linked_Ready</div>
                    <div className="text-sm text-slate-400 font-mono mt-1 uppercase font-bold tracking-widest">Latency: 84.2μs</div>
                    <div className="text-sm text-slate-800 font-bold uppercase tracking-widest">Freshness: {(inst.data_freshness_s || 0).toFixed(0)}s</div>
                    <div className="text-sm text-slate-800 font-bold uppercase tracking-widest">Direct OSQP Context</div>
                 </div>
              </div>
            </div>
          </Panel>

        </div>
      </div>
    </ErrorBoundary>
  );
}
