import React, { useState, useEffect, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import { 
  Activity, ShieldAlert, TrendingUp, Zap, Terminal, Cpu, Gauge, AlertTriangle,
  HelpCircle, X, Filter, GripVertical, Info
} from 'lucide-react';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, AreaChart, Area, ReferenceLine } from 'recharts';

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError(error) { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-full bg-[#050505] text-red-500 p-10 font-mono flex flex-col items-center justify-center border-[20px] border-red-900/20 shadow-inner">
          <ShieldAlert className="w-16 h-16 mb-4 animate-pulse" />
          <h1 className="text-2xl font-bold uppercase tracking-widest text-red-600">Terminal Kernel Panic</h1>
          <button onClick={() => window.location.reload()} className="mt-8 bg-red-600 text-white px-12 py-3 font-black uppercase text-xs hover:bg-red-500 shadow-2xl transition-all border-none">Emergency Reboot</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- Components ---

const TICKER_NAMES = {
  "SPY": "S&P 500 ETF Trust",
  "AAPL": "Apple Inc.",
  "MSFT": "Microsoft Corp.",
  "NVDA": "NVIDIA Corp.",
  "GOOGL": "Alphabet Inc. (Cl A)",
  "AMZN": "Amazon.com Inc.",
  "META": "Meta Platforms Inc.",
  "TSLA": "Tesla Inc.",
  "LLY": "Eli Lilly & Co.",
  "UNH": "UnitedHealth Group",
  "JPM": "JPMorgan Chase & Co.",
  "V": "Visa Inc.",
  "MA": "Mastercard Inc.",
  "AVGO": "Broadcom Inc.",
  "HD": "Home Depot Inc.",
  "PG": "Procter & Gamble",
  "COST": "Costco Wholesale",
  "JNJ": "Johnson & Johnson",
  "ABBV": "AbbVie Inc.",
  "MRK": "Merck & Co.",
  "BAC": "Bank of America",
  "CRM": "Salesforce Inc.",
  "ORCL": "Oracle Corp.",
  "ADBE": "Adobe Inc.",
  "AMD": "Advanced Micro Devices",
  "PEP": "PepsiCo Inc.",
  "KO": "Coca-Cola Co.",
  "TMO": "Thermo Fisher Scientific",
  "WMT": "Walmart Inc.",
  "MCD": "McDonald's Corp.",
  "CSCO": "Cisco Systems",
  "NFLX": "Netflix Inc.",
  "ABT": "Abbott Laboratories",
  "DHR": "Danaher Corp.",
  "WFC": "Wells Fargo & Co.",
  "ACN": "Accenture plc",
  "QCOM": "Qualcomm Inc.",
  "LIN": "Linde plc",
  "GE": "General Electric",
  "PM": "Philip Morris Int.",
  "TXN": "Texas Instruments",
  "INTU": "Intuit Inc.",
  "AMGN": "Amgen Inc.",
  "VZ": "Verizon Communications",
  "AMAT": "Applied Materials",
  "UNP": "Union Pacific Corp.",
  "LOW": "Lowe's Companies",
  "BX": "Blackstone Inc.",
  "GS": "Goldman Sachs Group",
  "ISRG": "Intuitive Surgical",
  "HON": "Honeywell Int.",
  "MS": "Morgan Stanley",
  "CVS": "CVS Health Corp.",
  "COP": "ConocoPhillips",
  "IBM": "IBM Corp.",
  "BA": "Boeing Co.",
  "SPGI": "S&P Global Inc.",
  "CAT": "Caterpillar Inc.",
  "LMT": "Lockheed Martin",
  "RTX": "Raytheon Technologies"
};

const PriceChart = ({ data, ticker }) => {
  const chartContainerRef = useRef(); 
  const chartRef = useRef(); 
  const seriesRef = useRef();
  const [range, setRange] = useState('ALL');
  const ranges = [
    { label: '1D', val: 24 * 3600 },
    { label: '1W', val: 7 * 24 * 3600 },
    { label: '1M', val: 30 * 24 * 3600 },
    { label: '3M', val: 90 * 24 * 3600 },
    { label: '1Y', val: 365 * 24 * 3600 },
    { label: 'ALL', val: null }
  ];

  const cleanData = React.useMemo(() => {
    if (!data) return [];
    const seen = new Set();
    return data.filter(d => !seen.has(d.time) && seen.add(d.time)).sort((a, b) => a.time - b.time);
  }, [data]);

  const rangeReturn = React.useMemo(() => {
    if (cleanData.length < 2) return 0;
    const last = cleanData[cleanData.length - 1].close;
    let first = cleanData[0].close;
    if (range !== 'ALL') {
        const r = ranges.find(x => x.label === range);
        const startTime = cleanData[cleanData.length - 1].time - r.val;
        const firstBar = cleanData.find(d => d.time >= startTime);
        if (firstBar) first = firstBar.close;
    }
    return ((last / first) - 1) * 100;
  }, [cleanData, range]);

  useEffect(() => {
    if (!chartContainerRef.current) return;
    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      width: chartContainerRef.current.clientWidth, height: 220,
      timeScale: { borderColor: '#1e293b', timeVisible: true, secondsVisible: false },
    });
    const series = chart.addSeries(CandlestickSeries, { upColor: '#10b981', downColor: '#ef4444', borderVisible: false });
    chartRef.current = chart; seriesRef.current = series;
    
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (seriesRef.current && cleanData.length > 0) {
      seriesRef.current.setData(cleanData);
      if (range === 'ALL') chartRef.current.timeScale().fitContent();
      else {
          const r = ranges.find(x => x.label === range);
          const last = cleanData[cleanData.length - 1].time;
          chartRef.current.timeScale().setVisibleRange({ from: last - r.val, to: last });
      }
    }
  }, [cleanData, range]); 

  return (
    <div className="w-full flex flex-col gap-1 font-mono">
      <div className="flex justify-between items-center mb-1 border-b border-slate-800/60 pb-1">
          <div className="flex items-center gap-2">
            <div className="text-[10px] text-emerald-500 uppercase font-black tracking-[0.2em]">{ticker} FEED</div>
            <div className={`text-[9px] font-bold px-1.5 rounded-sm ${rangeReturn >= 0 ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'}`}>
                {range}: {rangeReturn >= 0 ? '+' : ''}{rangeReturn.toFixed(2)}%
            </div>
          </div>
          <div className="flex gap-1">
            {ranges.map(r => (
              <button key={r.label} onClick={() => setRange(r.label)} className={`px-1.5 py-0.5 text-[8px] font-black border transition-all ${range === r.label ? 'bg-emerald-500 text-black border-emerald-500' : 'border-slate-800 text-slate-400 hover:text-emerald-400'}`}>{r.label}</button>
            ))}
          </div>
      </div>
      <div ref={chartContainerRef} className="w-full h-[220px] bg-black shadow-2xl border border-slate-800/60" />
    </div>
  );
};

const BenchmarkChart = ({ history }) => {
  if (!history || history.length < 2) return <div className="h-full w-full flex items-center justify-center text-slate-500 text-[10px] uppercase tracking-widest italic font-black">Awaiting Benchmarking...</div>;
  
  const initialPort = history[0].portfolio || 1;
  const initialSpy = history[0].spy || 1;
  
  // SENIOR FIX: Do not normalize to 100k. Show actual raw Net Liq to match the top bar.
  // The percentage return is still calculated correctly from the start of the window.
  const chartData = history.map(d => ({
    time: d.time,
    portfolio: d.portfolio,
    spy: d.spy,
    portPct: ((d.portfolio / initialPort) - 1) * 100,
    spyPct: ((d.spy / initialSpy) - 1) * 100
  }));

  const latestPort = chartData[chartData.length - 1].portfolio;
  const latestSpy = chartData[chartData.length - 1].spy;
  const latestPortPct = chartData[chartData.length - 1].portPct;
  const latestSpyPct = chartData[chartData.length - 1].spyPct;

  return (
    <div className="w-full h-full flex flex-col font-mono">
        <div className="flex justify-between items-center mb-3">
            <div className="flex gap-4">
                <div className="flex flex-col">
                  <span className="text-[8px] text-slate-400 uppercase font-black tracking-widest font-bold">Agent</span>
                  <span className={`text-xs font-black tabular-nums ${latestPort >= initialPort ? 'text-emerald-400' : 'text-rose-500'}`}>
                    ${(latestPort/1000).toFixed(1)}k ({latestPortPct >= 0 ? '+' : ''}{latestPortPct.toFixed(2)}%)
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[8px] text-slate-400 uppercase font-black tracking-widest font-bold">S&P 500</span>
                  <span className={`text-xs font-black tabular-nums ${latestSpy >= initialSpy ? 'text-emerald-400' : 'text-rose-500'}`}>
                    ${(latestSpy/1000).toFixed(1)}k ({latestSpyPct >= 0 ? '+' : ''}{latestSpyPct.toFixed(2)}%)
                  </span>
                </div>
            </div>
            <div className={`text-[10px] font-black px-2 py-0.5 border ${latestPortPct > latestSpyPct ? 'border-emerald-500/50 text-emerald-500 bg-emerald-500/5' : 'border-rose-500/50 text-rose-500'}`}>
                {latestPortPct > latestSpyPct ? 'ALPHA ACTIVE' : 'INDEX DOMINANT'}
            </div>
        </div>
        <div className="h-[180px] w-full bg-black/40 border border-slate-800/60 p-2 shadow-inner">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                    <defs>
                        <linearGradient id="colorPort" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/><stop offset="95%" stopColor="#10b981" stopOpacity={0}/></linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis 
                        dataKey="time" 
                        stroke="#cbd5e1" 
                        fontSize={8} 
                        tickFormatter={(str) => {
                          const parts = str.split("-");
                          return parts.length > 2 ? `${parts[1]}/${parts[2]}` : str;
                        }} 
                        minTickGap={30}
                    />
                    <YAxis 
                        domain={['auto', 'auto']} 
                        stroke="#cbd5e1" 
                        fontSize={8} 
                        tickFormatter={(val) => "$" + (val/1000).toFixed(0) + "k"} 
                    />
                    <Tooltip 
                        contentStyle={{backgroundColor: '#000', border: '1px solid #1e293b', fontSize: '10px'}} 
                        formatter={(val) => [`$${val.toLocaleString(undefined, {maximumFractionDigits:0})}`, 'Value']}
                    />
                    <Area type="monotone" dataKey="portfolio" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorPort)" dot={false} />
                    <Area type="monotone" dataKey="spy" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" fill="transparent" dot={false} />
                    <ReferenceLine y={initialPort} stroke="#475569" strokeWidth={1} strokeDasharray="3 3" label={{ position: 'right', value: 'Start', fill: '#475569', fontSize: 8, fontWeight: 'bold' }} />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    </div>
  );
};

const Heatmap = ({ data, title }) => {
  const canvasRef = useRef();
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current; const ctx = canvas.getContext('2d');
    const rows = data.length; const cols = data[0].length; canvas.width = cols; canvas.height = rows;
    const imageData = ctx.createImageData(cols, rows); const flatData = data.flat(); const maxVal = Math.max(...flatData.slice(0, 10000), 0.000001);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = data[i][j]; const ratio = Math.min(1.0, val / maxVal); const idx = (i * cols + j) * 4;
        imageData.data[idx] = ratio * 400; imageData.data[idx+1] = (ratio - 0.2) * 500; imageData.data[idx+2] = (ratio - 0.5) * 600; imageData.data[idx+3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [data]);
  return (
    <div className="w-full h-full flex flex-col min-h-0">
      <div className="text-[9px] font-black text-slate-400 mb-1 uppercase tracking-widest">{title}</div>
      <div className="h-64 relative border border-slate-800/60 bg-black overflow-hidden shadow-2xl"><canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ imageRendering: 'pixelated' }} /></div>
    </div>
  );
};

const SortableRow = ({ row, onSelectTicker }) => {
  const [flash, setFlash] = useState(null);
  const prevPriceRef = useRef(row.live_price);
  useEffect(() => { if (row.live_price !== prevPriceRef.current) { setFlash(row.live_price > prevPriceRef.current ? 'up' : 'down'); setTimeout(() => setFlash(null), 800); prevPriceRef.current = row.live_price; } }, [row.live_price]);
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: row.ticker });
  const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1, zIndex: isDragging ? 50 : 1, backgroundColor: flash === 'up' ? 'rgba(16, 185, 129, 0.1)' : flash === 'down' ? 'rgba(239, 68, 68, 0.1)' : 'transparent' };
  return (
    <tr ref={setNodeRef} style={style} className="border-b border-slate-800/40 hover:bg-slate-800/60 cursor-pointer group font-mono tabular-nums transition-colors text-[11px]">
      <td className="py-1.5 pl-2 w-8"><div {...attributes} {...listeners} className="cursor-grab p-1 text-slate-500 hover:text-emerald-500"><GripVertical className="w-3 h-3" /></div></td>
      <td onClick={() => onSelectTicker(row.ticker)} className="py-1.5 flex items-center gap-2"><div className={"w-1 h-3 " + (row.score > 0 ? "bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.5)]" : "bg-rose-500")} /><span className="font-black text-slate-100 uppercase">{row.ticker}</span></td>
      <td onClick={() => onSelectTicker(row.ticker)} className="text-right py-1.5 text-slate-300 pr-4 font-bold font-mono text-xs tracking-tighter">${(row.live_price || 0).toFixed(2)}</td>
      <td onClick={() => onSelectTicker(row.ticker)} className={"text-right py-1.5 pr-4 font-bold font-mono text-xs tracking-tighter " + (row.score > 0 ? "text-emerald-500" : "text-rose-500")}>{(row.score || 0).toFixed(4)}</td>
      <td onClick={() => onSelectTicker(row.ticker)} className="text-right py-1 pr-4">
        {Math.abs(row.market_value) > 0 ? (
          <div className="flex flex-col text-right leading-tight font-mono">
            <div className="flex justify-end items-baseline gap-1">
              <span className="font-black text-slate-100 text-xs">${row.market_value.toLocaleString(undefined, {maximumFractionDigits:0})}</span>
              <span className="text-[8px] text-slate-500 font-bold uppercase">x{row.qty}</span>
            </div>
            <span className={`text-[9px] font-black ${row.pnl_pct >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
              ({row.pnl_pct >= 0 ? '+' : ''}{row.pnl_pct.toFixed(2)}%)
            </span>
          </div>
        ) : <span className="text-slate-700">—</span>}
      </td>
      <td onClick={() => onSelectTicker(row.ticker)} className="text-right py-1.5 pr-2"><span className={"text-[9px] px-1.5 py-0.5 rounded-sm border uppercase font-black tracking-widest " + (row.score > 0 ? "border-emerald-900/50 text-emerald-500 bg-emerald-500/5" : "border-rose-900/50 text-rose-500 bg-rose-500/5")}>{row.score > 0 ? "Long" : "Short"}</span></td>
    </tr>
  );
};

const RankingGrid = ({ ladder, onSelectTicker, filterSector, tickerOrder, sensors, handleDragEnd }) => {
  const [showHoldingsOnly, setShowHoldingsOnly] = useState(false);
  const sortedData = React.useMemo(() => {
    if (!ladder || !tickerOrder) return [];
    const map = {}; ladder.forEach(item => { map[item.ticker] = item; });
    let items = tickerOrder.map(ticker => map[ticker]).filter(item => item !== undefined);
    if (showHoldingsOnly) items = items.filter(item => Math.abs(item.market_value) > 0.1);
    if (filterSector) items = items.filter(item => item.sector === filterSector);
    return items;
  }, [ladder, tickerOrder, filterSector, showHoldingsOnly]);
  return (
    <div className="h-[520px] overflow-y-auto no-scrollbar pr-2 mb-4">
      <div className="flex justify-between items-center mb-3 border-b border-slate-800 pb-2">
        <div className="text-[10px] font-black text-slate-400 uppercase tracking-[0.4em]">Decile Ladder</div>
        <button onClick={() => setShowHoldingsOnly(!showHoldingsOnly)} className={`px-2 py-0.5 text-[9px] font-black uppercase border transition-all ${showHoldingsOnly ? 'bg-emerald-600 text-black border-emerald-600 shadow-[0_0_10px_rgba(16,185,129,0.3)]' : 'border-slate-800 text-slate-400 hover:text-white'}`}>Holdings</button>
      </div>
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <table className="w-full text-[11px] border-collapse font-mono shadow-2xl">
          <thead className="sticky top-0 bg-slate-950 text-slate-400 uppercase z-10 border-b border-slate-800 shadow-sm font-bold">
            <tr><th className="w-8"></th><th className="text-left py-2 tracking-widest">Ticker</th><th className="text-right py-2 pr-4 tracking-widest">Price</th><th className="text-right py-2 pr-4 tracking-widest">Score</th><th className="text-right py-2 pr-4 tracking-widest">Position</th><th className="text-right py-2 pr-2 tracking-widest font-bold">Action</th></tr>
          </thead>
          <SortableContext items={tickerOrder} strategy={verticalListSortingStrategy}>
            <tbody>{sortedData.map((row) => (<SortableRow key={row.ticker} row={row} onSelectTicker={onSelectTicker} />))}</tbody>
          </SortableContext>
        </table>
      </DndContext>
    </div>
  );
};

const Panel = ({ title, icon: Icon, children, className = "" }) => (
  <div className={"bg-black border border-slate-800/60 p-4 flex flex-col h-full shadow-2xl backdrop-blur-md " + className}>
    <div className="flex items-center gap-2 mb-3 border-b border-slate-800/60 pb-2"><Icon className="w-3.5 h-3.5 text-emerald-500/80" /><h2 className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-400">{title}</h2></div>
    <div className="flex-1">{children}</div>
  </div>
);

const MissionManual = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-xl p-4 sm:p-20 font-mono">
      <div className="bg-[#0a0a0a] border border-emerald-500/20 w-full max-w-4xl shadow-[0_0_100px_rgba(16,185,129,0.1)] relative flex flex-col max-h-full">
        <div className="flex justify-between items-center p-6 border-b border-emerald-500/10">
          <div className="flex items-center gap-3">
            <Terminal className="w-5 h-5 text-emerald-500" />
            <h2 className="text-emerald-500 font-black uppercase tracking-[0.3em] text-sm">Institutional Operations Manual</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-emerald-500/10 text-emerald-500 transition-all border border-transparent hover:border-emerald-500/20"><X className="w-5 h-5" /></button>
        </div>
        <div className="p-8 space-y-6 text-xs text-slate-300 leading-relaxed overflow-y-auto">
          <section>
            <h3 className="font-bold text-emerald-500 uppercase tracking-widest mb-3 border-b border-emerald-500/10 pb-1">1. Spectral Alpha Engine</h3>
            <p>Visualizes multi-scale wavelet energy distribution. Click any ticker in the Ranking Ladder to hydrate the technical feed. The ADF p-value indicates time-series stationarity (Target: &lt; 0.05). Heatmap represents the Continuous Wavelet Transform (CWT) spectrogram.</p>
          </section>
          <section>
            <h3 className="font-bold text-emerald-500 uppercase tracking-widest mb-3 border-b border-emerald-500/10 pb-1">2. RL Metacognition</h3>
            <p>Displays the internal state of the PPO Policy Pilot. **Policy Conviction** tracks the agent's mathematical expectation of alpha. **Manifold Drift** (t-SNE Latent) detects structural regime shifts by projecting high-dimensional feature space into a 2D drift plane.</p>
          </section>
          <section>
            <h3 className="font-bold text-emerald-500 uppercase tracking-widest mb-3 border-b border-emerald-500/10 pb-1">3. Execution Muscle (C++26)</h3>
            <p>High-frequency OMS kernel running Quadratic Slippage models. All fills are simulated with a 15bps realistic institutional tax. The **Implementation Shortfall** metric monitors the gap between decision price and fill price.</p>
          </section>
          <section>
            <h3 className="font-bold text-emerald-500 uppercase tracking-widest mb-3 border-b border-emerald-500/10 pb-1">4. Performance Benchmarking</h3>
            <p>Real-time attribution against the S&P 500. Portfolio value is standardized to $100k at start of simulation. The **Alpha Active** indicator triggers when the Agent Portfolio out-performs the market on a total return basis.</p>
          </section>
        </div>
        <div className="p-6 border-t border-emerald-500/10 text-[9px] text-emerald-900 font-black uppercase tracking-widest text-center italic">uqts_v4.1_elite_hybrid // secure_terminal_session</div>
      </div>
    </div>
  );
};

const ShapChart = ({ data }) => {
  if (!data) return null;
  const chartData = Object.entries(data).map(([name, value]) => ({ name: name.split(' ')[0], value }));
  return (
    <div className="flex-none w-full mt-4 border-t border-slate-800/60 pt-4">
      <div className="text-[10px] text-slate-300 font-bold uppercase tracking-widest mb-2">Signal Fusion (SHAP)</div>
      <div className="h-[180px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical">
            <XAxis type="number" hide />
            <YAxis dataKey="name" type="category" stroke="#cbd5e1" fontSize={8} width={50} />
            <Tooltip 
              contentStyle={{backgroundColor: '#000', border: '1px solid #1e293b', fontSize: '10px'}} 
              cursor={{fill: 'rgba(16, 185, 129, 0.05)'}}
            />
            <Bar dataKey="value" fill="#10b981" radius={[0, 2, 2, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const AlphaGainChart = ({ data }) => {
  const [range, setRange] = useState('ALL');
  const ranges = [{ label: '1W', val: 7 }, { label: '1M', val: 30 }, { label: '3M', val: 90 }, { label: '1Y', val: 365 }, { label: 'ALL', val: null }];
  
  const { filteredData, rangeAlpha, splitOffset } = React.useMemo(() => {
    if (!data || data.length === 0) return { filteredData: [], rangeAlpha: 0, splitOffset: 0 };
    
    let slice = data;
    if (range !== 'ALL') {
      const r = ranges.find(x => x.label === range);
      slice = data.slice(-r.val);
    }
    
    if (slice.length < 2) return { filteredData: slice, rangeAlpha: 0, splitOffset: 0 };
    
    const latest = slice[slice.length - 1].alpha;
    const start = slice[0].alpha;
    
    // Calculate precise gradient offset for the 0.0 line
    const alphas = slice.map(d => d.alpha);
    const dataMax = Math.max(...alphas);
    const dataMin = Math.min(...alphas);
    
    let offset = 0;
    if (dataMax <= 0) offset = 0; // Entirely below zero
    else if (dataMin >= 0) offset = 1; // Entirely above zero
    else offset = dataMax / (dataMax - dataMin);

    return { filteredData: slice, rangeAlpha: latest - start, splitOffset: offset };
  }, [data, range]);

  return (
    <div className="flex-none flex flex-col mt-4 border-t border-slate-800/60 pt-4">
      <div className="flex justify-between items-center mb-2">
        <div className="flex flex-col">
          <div className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
            Alpha Velocity :: <span className={rangeAlpha >= 0 ? 'text-emerald-400' : 'text-rose-500'}>
                {rangeAlpha >= 0 ? '+' : ''}{rangeAlpha.toFixed(2)}% {range}
            </span>
          </div>
        </div>
        <div className="flex gap-1">
          {ranges.map(r => (
            <button key={r.label} onClick={() => setRange(r.label)} className={`px-1.5 py-0.5 text-[7px] font-black border transition-all ${range === r.label ? 'bg-emerald-500 text-black border-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.3)]' : 'border-slate-800 text-slate-500 hover:text-white'}`}>{r.label}</button>
          ))}
        </div>
      </div>
      <div className="h-[180px] w-full bg-black/40 border border-slate-800/60 p-2 shadow-inner">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={filteredData}>
            <defs>
                <linearGradient id="splitFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={splitOffset} stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset={splitOffset} stopColor="#ef4444" stopOpacity={0.3} />
                </linearGradient>
                <linearGradient id="splitStroke" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={splitOffset} stopColor="#10b981" stopOpacity={1} />
                    <stop offset={splitOffset} stopColor="#ef4444" stopOpacity={1} />
                </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="1 12" stroke="#1e293b" vertical={false} />
            <XAxis dataKey="time" hide />
            <YAxis domain={['auto', 'auto']} stroke="#cbd5e1" fontSize={8} tickFormatter={(v) => v.toFixed(1) + "%"} />
            <Tooltip contentStyle={{backgroundColor: '#000', border: '1px solid #1e293b', fontSize: '10px'}} formatter={(val) => [val.toFixed(2) + "%", 'Cumulative Alpha']} />
            <Area type="monotone" dataKey="alpha" stroke="url(#splitStroke)" fill="url(#splitFill)" dot={false} strokeWidth={3} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const TelemetryFailsafe = () => (
  <div className="h-full w-full flex flex-col items-center justify-center text-emerald-500/20 font-mono gap-3 animate-pulse uppercase tracking-[0.3em] text-[9px] py-20">
    <Zap className="w-6 h-6 opacity-20" />
    <span>Awaiting Telemetry...</span>
  </div>
);

export default function MissionControl() {
  const [globalData, setGlobalData] = useState(null);
  const [spectralData, setSpectralData] = useState(null);
  const [status, setStatus] = useState('connecting');
  const [tick, setTick] = useState(0);
  const [selectedSector, setSelectedSector] = useState(null);
  const [investedOnly, setInvestedOnly] = useState(false);
  const [isManualOpen, setIsManualOpen] = useState(false);
  const [tickerOrder, setTickerOrder] = useState([]);
  
  const ws = useRef(null);
  const lastMsgRef = useRef(Date.now());
  const sensors = useSensors(useSensor(PointerSensor), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));

  const connect = () => {
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) return;
    
    setStatus(prev => prev !== 'connecting' ? 'connecting' : prev);
    const socket = new WebSocket('ws://localhost:8000/ws/cockpit');
    
    socket.onopen = () => {
      setStatus('active');
      lastMsgRef.current = Date.now();
      socket.send(JSON.stringify({ command: 'SET_TICKER', ticker: 'SPY' }));
    };

    socket.onmessage = (e) => {
      lastMsgRef.current = Date.now();
      try {
        const payload = JSON.parse(e.data);
        if (payload.type === 'GLOBAL_UPDATE') {
          setGlobalData(payload);
          if (payload.rankings?.ladder && tickerOrder.length === 0) {
            setTickerOrder(payload.rankings.ladder.map(item => item.ticker).sort((a, b) => a.localeCompare(b)));
          }
        } else if (payload.type === 'SPECTRAL_UPDATE') {
          setSpectralData(payload.spectral);
        }
      } catch (err) {
        console.error("Payload parse error", err);
      }
    };

    socket.onclose = () => setStatus('disconnected');
    socket.onerror = () => setStatus('disconnected');
    ws.current = socket;
  };

  useEffect(() => {
    connect();
    const heartbeat = setInterval(() => {
      setTick(t => t + 1);
      const diff = Date.now() - lastMsgRef.current;
      
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        setStatus(prev => prev !== 'connecting' ? 'disconnected' : 'connecting');
      } else if (diff > 10000) {
        setStatus('disconnected');
      } else if (diff > 5000) {
        setStatus('stale');
      } else {
        setStatus('active');
      }
    }, 1000);

    const autoReconnect = setInterval(() => {
      if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
        connect();
      }
    }, 3000);

    return () => {
      clearInterval(heartbeat);
      clearInterval(autoReconnect);
      ws.current?.close();
    };
  }, []); // EMPTY dependency array is critical

  const handleDragEnd = (event) => {
    const { active, over } = event;
    if (active && over && active.id !== over.id) {
      setTickerOrder((items) => {
        const oldIndex = items.indexOf(active.id);
        const newIndex = items.indexOf(over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  const handleSelectTicker = (ticker) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: 'SET_TICKER', ticker }));
    }
  };

  const inst = globalData?.institutional || {};
  const meta = globalData?.metacognition || {};
  const exec = globalData?.execution || {};
  const pipeline = globalData?.pipeline || {};
  const strat = meta.strategy_sensors || {};

  return (
    <div className="min-h-screen w-full bg-[#050505] text-slate-300 font-mono text-[13px] flex flex-col p-2 gap-2 overflow-x-hidden selection:bg-emerald-500/30">
      <MissionManual isOpen={isManualOpen} onClose={() => setIsManualOpen(false)} />
      
      <div className="flex-none h-14 flex justify-between items-center border-b border-slate-800/60 pb-2 px-2 bg-black/60 shadow-2xl">
        <div className="flex items-center gap-4 h-full">
          <div className="text-xl font-black text-white tracking-tighter italic border-r border-slate-800/60 pr-6 mr-2 h-full flex items-center uppercase">UQTS-2026</div>
          <div className="flex items-center gap-6 h-full font-mono">
             <div className="flex items-center gap-4 bg-slate-900/30 px-3 py-1 rounded-sm border border-slate-800/40 shadow-inner">
                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${
                      status === "active" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse" : 
                      status === "stale" ? "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]" : 
                      status === "connecting" ? "bg-cyan-500 animate-bounce" : "bg-red-500"
                    }`} />
                    <span className="uppercase text-[9px] font-black text-slate-400 tracking-widest">{status}</span>
                  </div>
                  <div className="border-l border-slate-800 h-3" />
                  <div className="flex items-center gap-1.5">
                    <Activity className="w-2.5 h-2.5 text-emerald-500" />
                    <span className="uppercase text-[9px] font-black text-slate-500 tabular-nums">
                      HB: {Math.max(0, Math.round((Date.now() - lastMsgRef.current) / 1000))}s
                    </span>
                  </div>
             </div>
             <div className="flex flex-col border-r border-slate-800/60 pr-6 mr-6">
                <span className="text-[8px] text-slate-400 uppercase font-black tracking-tighter tabular-nums font-bold">Event Horizon</span>
                <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-100 tabular-nums">{globalData?.timestamp || 'OFFLINE'}</span>
                    {globalData?.market_status && (
                        <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-[2px] border ${globalData.market_status === 'OPEN' ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.1)]' : 'border-rose-900/30 bg-rose-950/10 text-rose-500'}`}>
                            <div className={`w-1 h-1 rounded-full ${globalData.market_status === 'OPEN' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-900'}`} />
                            <span className="text-[8px] font-black tracking-widest">{globalData.market_status}</span>
                        </div>
                    )}
                </div>
             </div>
             <div className="flex gap-4 border-l border-slate-800/60 pl-6 h-full items-center">
                 <div className="flex flex-col text-center"><span className="text-[8px] text-slate-400 uppercase font-black tracking-tighter font-bold">Net Liq</span><span className="text-xs font-black text-emerald-500 tabular-nums">${(inst.capital || 0).toLocaleString()}</span></div>
                 <div className="flex flex-col border-l border-slate-800/40 pl-4 text-center">
                    <span className="text-[8px] text-slate-400 uppercase font-black tracking-tighter font-bold">
                        {inst.buying_power < 0 ? 'Margin' : 'BP'}
                    </span>
                    <span className={`text-xs font-black tabular-nums ${inst.buying_power < 0 ? 'text-rose-500' : 'text-cyan-500'}`}>
                        ${(inst.buying_power || 0).toLocaleString()}
                    </span>
                 </div>
                 <div className="flex flex-col border-l border-slate-800/40 pl-4 text-center"><span className="text-[8px] text-slate-400 uppercase font-black tracking-tighter uppercase font-bold">EXP</span><span className="text-xs font-black text-slate-100 tabular-nums">{(inst.gross_exposure || 0).toFixed(1)}%</span></div>
                 <div className="flex flex-col border-l border-slate-800/40 pl-4 text-center"><span className="text-[8px] text-slate-400 uppercase font-black tracking-tighter uppercase font-bold">ROE</span><span className={`text-xs font-black tabular-nums ${inst.roe >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>{inst.roe >= 0 ? '+' : ''}{(inst.roe || 0).toFixed(2)}%</span></div>
             </div>
             <div className="flex gap-4 border-l-2 border-emerald-500/40 pl-6 bg-emerald-500/5 px-4 h-full items-center rounded-r-md">
                <div className="flex flex-col text-center"><span className="text-[8px] text-emerald-600/80 uppercase font-black tracking-tighter italic">RL LEV</span><span className="text-xs font-black text-emerald-400 tabular-nums">{(meta.rl_leverage || 0).toFixed(2)}x</span></div>
                <div className="flex flex-col border-l border-emerald-900/40 pl-3 text-center"><span className="text-[8px] text-emerald-600/80 uppercase font-black tracking-tighter italic text-[7px]">Hedge</span><span className={`text-[10px] font-black tabular-nums italic ${meta.rl_hedge > 0 ? 'text-amber-500' : 'text-slate-500'}`}>{meta.rl_hedge > 0 ? `SH ${(meta.rl_hedge * 100).toFixed(0)}%` : 'OFF'}</span></div>
                <div className="flex flex-col border-l border-emerald-900/40 pl-3 text-center"><span className="text-[8px] text-emerald-600/80 uppercase font-black tracking-tighter italic text-[7px]">Conc</span><span className="text-xs font-black text-emerald-400 italic">T{meta.concentration || 5}</span></div>
             </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button onClick={() => setIsManualOpen(true)} className="flex items-center gap-2 px-3 py-1 text-emerald-500/60 hover:text-emerald-400 border border-emerald-500/20 hover:border-emerald-500/40 transition-all uppercase text-[9px] font-black tracking-widest"><HelpCircle className="w-3.5 h-3.5" /> Manual</button>
          <button onClick={() => ws.current?.send(JSON.stringify({command: 'KILL_SWITCH'}))} className="px-5 py-1 transition-all font-black uppercase text-[10px] border border-rose-900 bg-rose-950/10 text-rose-600 hover:bg-rose-600 hover:text-white rounded-sm tracking-widest ml-auto">Kill_Sys</button>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-2 pb-10">
        <Panel 
          title={`SPECTRAL :: ${TICKER_NAMES[spectralData?.ticker] || spectralData?.ticker || '---'} [${spectralData?.adf_p_value < 0.05 ? 'LOCKED' : 'DRIFT'}]`} 
          icon={Activity} 
          className="col-span-12 xl:col-span-4 shadow-2xl"
        >
          <div className="flex flex-1 flex-col gap-4 min-h-0">
            {spectralData && spectralData.history ? (
              <>
                <div className="flex-none"><PriceChart data={spectralData.history} ticker={spectralData.ticker} /></div>
                <div className="flex-none relative shadow-2xl border border-slate-800/60 bg-black">
                  <Heatmap data={spectralData.cwt} title={spectralData.ticker + " CWT Spectrogram"} />
                </div>
                <div className="grid grid-cols-2 gap-1.5 flex-none">
                  <div className="bg-slate-900/40 p-2 border border-slate-800/60 text-center shadow-inner">
                    <div className="text-[10px] text-slate-400 uppercase font-black mb-1 tracking-widest font-bold">ADF p-value</div>
                    <div className={"text-lg font-mono tabular-nums " + (spectralData.adf_p_value < 0.05 ? "text-emerald-500" : "text-slate-400")}>
                      {(spectralData.adf_p_value || 0.0001).toFixed(6)}
                    </div>
                  </div>
                  <div className="bg-slate-900/40 p-2 border border-slate-800/60 text-center flex flex-col items-center justify-center shadow-inner">
                    <div className="text-[10px] text-slate-400 uppercase font-black mb-1 font-bold">Alpha State</div>
                    <span className="text-emerald-500 text-[10px] font-black border border-emerald-900/50 px-2 py-0.5 rounded-sm bg-emerald-500/5">READY_LOCK</span>
                  </div>
                </div>
                <ShapChart data={spectralData.shap_values} />
              </>
            ) : (
              <div className="h-[330px] w-full flex items-center justify-center text-slate-500 italic uppercase tracking-[0.3em] border border-slate-900/60 bg-black">
                {status === 'active' ? "Select focus ticker" : "Awaiting Telemetry..."}
              </div>
            )}
          </div>
        </Panel>

        <Panel 
          title={`INTELLIGENCE :: ${((meta.policy_conviction || 0) * 100).toFixed(2)}% CONVICTION`} 
          icon={ShieldAlert} 
          className="col-span-12 xl:col-span-4 shadow-2xl"
        >
          {!globalData ? <TelemetryFailsafe /> : (
            <div className="flex flex-1 flex-col gap-6 min-h-0">
               <div className="flex-none border border-slate-800/60 bg-black p-4 relative shadow-inner">
                 <div className="text-[9px] text-slate-400 mb-4 uppercase font-black text-center tracking-[0.4em] font-bold italic">Performance Benchmarking ($100k)</div>
                 <BenchmarkChart history={inst.performance_history} />
               </div>
               <div className="flex-none flex justify-between items-center bg-emerald-500/5 p-4 border border-emerald-900/30 shadow-xl">
                 <div className="flex flex-col text-left">
                   <span className="text-[10px] text-slate-300 uppercase font-black tracking-widest italic">Policy Conviction</span>
                   <span className="text-4xl font-black text-emerald-400 tabular-nums font-mono">{((meta.policy_conviction || 0) * 100).toFixed(2) + "%"}</span>
                 </div>
                 <Gauge className="w-10 h-10 text-emerald-500 opacity-20" />
               </div>
               <div className="flex-none grid grid-cols-3 gap-1.5">
                 <div className="bg-slate-900/40 p-3 border border-slate-800/60 text-center shadow-inner">
                   <div className="text-[8px] text-slate-300 uppercase font-black mb-0.5 font-bold">Win Rate</div>
                   <div className="text-xs font-black text-emerald-500 tabular-nums">{(strat.win_rate || 0).toFixed(1)}%</div>
                 </div>
                 <div className="bg-slate-900/40 p-3 border border-slate-800/60 text-center shadow-inner">
                   <div className="text-[8px] text-slate-300 uppercase font-black mb-0.5 font-bold">Max DD</div>
                   <div className="text-xs font-black text-rose-500 tabular-nums">{(strat.max_dd || 0).toFixed(2)}%</div>
                 </div>
                 <div className="bg-slate-900/40 p-3 border border-slate-800/60 text-center shadow-inner">
                   <div className="text-[8px] text-slate-300 uppercase font-black mb-0.5 font-bold">IC (Avg)</div>
                   <div className="text-xs font-black text-cyan-500 tabular-nums">{(strat.ic || 0).toFixed(4)}</div>
                 </div>
               </div>
               <AlphaGainChart data={meta.alpha_gain} />
               {inst.pending_decision && (
                    <div className="bg-emerald-500/5 border border-emerald-500/20 p-2 mt-4 rounded-sm animate-pulse">
                        <div className="text-[9px] font-black text-emerald-400 uppercase tracking-[0.2em] mb-2 flex justify-between border-b border-emerald-900/40 pb-1">
                            <span>🔮 Strategy Queue (Next Day)</span>
                            <span>Lev: {inst.pending_decision.target_lev}x</span>
                        </div>
                        
                        <div className="space-y-2">
                            <div className="flex flex-wrap gap-1">
                                <span className="text-[8px] text-slate-500 font-black w-8 uppercase pt-1">Picks:</span>
                                {inst.pending_decision.ladder?.map(x => (
                                    <span key={x.ticker} className="text-[9px] px-1.5 py-0.5 bg-black border border-emerald-900/50 text-emerald-500 font-bold uppercase">{x.ticker} ({x.qty})</span>
                                ))}
                            </div>

                            {inst.pending_decision.adds_display?.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                    <span className="text-[8px] text-emerald-600 font-black w-8 uppercase pt-1">Adds :</span>
                                    {inst.pending_decision.adds_display.map(str => (
                                        <span key={str} className="text-[9px] px-1.5 py-0.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold uppercase">{str}</span>
                                    ))}
                                </div>
                            )}

                            {inst.pending_decision.sells_display?.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                    <span className="text-[8px] text-rose-600 font-black w-8 uppercase pt-1">Sells:</span>
                                    {inst.pending_decision.sells_display.map(str => (
                                        <span key={str} className="text-[9px] px-1.5 py-0.5 bg-rose-500/10 border border-rose-500/30 text-rose-400 font-bold uppercase">{str}</span>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="text-[8px] text-slate-500 mt-2 italic font-bold border-t border-slate-900 pt-1">Planned at {inst.pending_decision.date} :: Dispatches at 15:50 EST</div>
                    </div>
                )}
            </div>
          )}
        </Panel>

        <Panel 
          title={`HIERARCHY :: ${inst.active_positions || 0} ACTIVE POS`} 
          icon={TrendingUp} 
          className="col-span-12 xl:col-span-4 shadow-2xl"
        >
           {!globalData ? <TelemetryFailsafe /> : (
             <>
               <RankingGrid ladder={globalData.rankings?.ladder} onSelectTicker={handleSelectTicker} filterSector={selectedSector} tickerOrder={tickerOrder} sensors={sensors} handleDragEnd={handleDragEnd} />
               <div className="flex-1 border-t border-slate-800/60 mt-4 pt-4 overflow-y-auto no-scrollbar shadow-inner">
                  <div className="text-[10px] text-slate-300 mb-2 uppercase font-black tracking-[0.3em] flex justify-between items-center pr-2">
                    <span className="font-bold">Sector Matrix</span>
                    <button onClick={() => setInvestedOnly(!investedOnly)} className={`text-[8px] px-2 py-0.5 border transition-all uppercase font-black ${investedOnly ? 'bg-emerald-600 text-black border-emerald-600 shadow-[0_0_10px_rgba(16,185,129,0.3)]' : 'border-slate-800 text-slate-400 hover:text-white'}`}>
                      {investedOnly ? "View: All Sectors" : "View: Invested Only"}
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 pb-4">
                    {Object.entries(inst.sector_exposure || {})
                        .filter(([_, stats]) => !investedOnly || stats.count > 0)
                        .map(([sector, stats]) => (
                         <button key={sector} onClick={() => setSelectedSector(selectedSector === sector ? null : sector)} className={"border p-1.5 flex flex-col justify-center transition-all text-left " + (selectedSector === sector ? "bg-emerald-500/10 border-emerald-500 shadow-xl" : "bg-slate-900/20 border-slate-800 hover:bg-slate-900/40")}>
                            <div className="text-[9px] text-slate-300 uppercase font-black truncate tracking-tighter">{sector}</div>
                            <div className={"text-sm font-black tabular-nums " + (stats.exposure > 0 ? "text-emerald-500" : "text-rose-500")}>{(stats.exposure > 0 ? "+" : "") + stats.exposure.toFixed(1)}%</div>
                            <div className="flex justify-between items-center mt-1 border-t border-slate-800/40 pt-1 font-black italic text-[8px] text-emerald-900/80 uppercase tracking-tighter">α: {(stats.avg_score || 0).toFixed(2)}</div>
                         </button>
                      ))}
                  </div>
               </div>
             </>
           )}
        </Panel>

        <Panel title="Execution Muscle" icon={Cpu} className="col-span-12 mt-2 shadow-2xl">
          {!globalData ? <TelemetryFailsafe /> : (
            <div className="grid grid-cols-12 gap-8 p-1">
              <div className="col-span-12 md:col-span-2 flex flex-col justify-center border-b md:border-b-0 md:border-r border-slate-800/60 pb-4 md:pb-0 md:pr-8">
                <div className="text-[10px] text-slate-400 uppercase font-black mb-1 tracking-widest font-bold">Imp Shortfall</div>
                <div className="text-3xl font-black text-slate-100 tabular-nums italic drop-shadow-[0_0_10px_rgba(255,255,255,0.1)]">{(exec.implementation_shortfall || 0).toFixed(2)} <span className="text-xs text-slate-500 not-italic uppercase font-bold tracking-tighter font-mono">BPS</span></div>
                
                <div className="mt-4 border-t border-slate-800/40 pt-4">
                    <div className="text-[10px] text-slate-400 uppercase font-black mb-1 tracking-widest font-bold">Total Fees</div>
                    <div className="text-xl font-black text-rose-500 tabular-nums italic shadow-sm">
                        ${(exec.cumulative_fees || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                    </div>
                </div>

                <div className="text-[9px] text-slate-800 font-black tracking-widest mt-4 uppercase italic font-mono opacity-40">muscle_kernel::C++26</div>
              </div>
              <div className="col-span-12 md:col-span-6 border-b md:border-b-0 md:border-r border-slate-800/60 pb-4 md:pb-0 md:pr-8 flex flex-col gap-2">
                <div className="text-[10px] text-slate-300 uppercase font-black flex justify-between items-center border-b border-slate-800/60 pb-1 tracking-widest">
                    <span className="font-bold">OMS Queue (Live)</span>
                    <div className="flex gap-4 font-bold tracking-tighter text-[9px] uppercase">
                      <span className="text-emerald-500 font-black">{(inst.oms_queue?.filled || 0)} Filled</span>
                      <span className="text-amber-500 font-black">{(inst.oms_queue?.working || 0)} Working</span>
                    </div>
                </div>
                <div className="h-40 overflow-y-auto no-scrollbar font-mono text-[11px] tabular-nums bg-black/20 p-1 shadow-inner">
                    {(inst.order_log || []).slice().reverse().map((log, i) => (
                        <div key={i} className="flex justify-between items-center border-b border-slate-900/50 pb-1.5 mb-1.5 hover:bg-emerald-500/5 transition-colors group">
                            <span className="text-slate-500 w-32 tracking-tighter font-bold group-hover:text-slate-300 text-[10px]">{log.time}</span>
                            <span className={`font-black w-24 ${log.side === 'BUY' ? 'text-emerald-500' : 'text-rose-500'}`}>{log.side + " " + log.ticker}</span>
                            <div className="flex items-center gap-4 flex-1 justify-end pr-8 font-mono">
                                <span className="text-slate-500 font-bold uppercase text-[9px] opacity-60 tracking-tighter">QTY: {log.qty}</span>
                                <span className="text-emerald-500 font-black tracking-tighter italic shadow-sm">${(log.notional || 0).toLocaleString()}</span>
                            </div>
                            <span className="text-emerald-500 font-black uppercase text-[8px] tracking-[0.2em] bg-emerald-500/5 px-2 py-0.5 border border-emerald-900/30 rounded-sm shadow-inner">Filled</span>
                        </div>
                    ))}
                    {(!inst.order_log || inst.order_log.length === 0) && <div className="h-full flex items-center justify-center text-slate-800 italic uppercase tracking-[0.4em] animate-pulse text-[10px] font-black">Awaiting instructions...</div>}
                </div>
              </div>
              <div className="col-span-12 md:col-span-2 border-b md:border-b-0 md:border-r border-slate-800/60 pb-4 md:pb-0 md:pr-8 flex flex-col justify-center gap-2">
                    <div className="text-[10px] text-slate-400 uppercase font-black text-center tracking-[0.3em] font-bold uppercase font-mono">Slippage Matrix</div>
                    <div className="grid grid-cols-5 gap-1 bg-black p-2 border border-slate-800/60 h-24 shadow-inner">
                      {(exec.slippage_heatmap || []).flat().map((v, i) => (
                        <div key={i} className="bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.1)] transition-all duration-1000" style={{ opacity: v }} />
                      ))}
                    </div>
              </div>
              <div className="col-span-12 md:col-span-2 flex flex-col justify-center items-end text-right gap-4">
                  <div className="grid grid-cols-2 gap-2 font-black uppercase tracking-tighter w-full">
                      <div className="bg-slate-900/60 p-3 border border-slate-700 flex flex-col items-center shadow-2xl rounded-sm">
                        <span className="text-[10px] text-slate-200 uppercase tracking-widest font-bold mb-1">Champ</span>
                        <span className="text-xl text-cyan-400 font-black font-mono tracking-tighter">{(pipeline.champion_sharpe || 1.15).toFixed(2)}</span>
                      </div>
                      <div className="bg-slate-900/60 p-3 border border-slate-700 flex flex-col items-center shadow-2xl rounded-sm">
                        <span className="text-[10px] text-slate-200 uppercase tracking-widest font-bold mb-1">Chal</span>
                        <span className="text-xl text-emerald-400 font-black font-mono tracking-tighter">{(pipeline.challenger_sharpe || 0).toFixed(2)}</span>
                      </div>
                  </div>
                  <div className="flex flex-col text-right font-mono">
                    <span className="text-[11px] text-slate-400 font-black uppercase tracking-[0.2em] truncate">Kernel_v4.1</span>
                    <span className="text-[11px] text-emerald-500/80 font-black uppercase tracking-widest">
                      Fresh: {Math.round((Date.now() - lastMsgRef.current) / 1000)}s
                    </span>
                  </div>
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
