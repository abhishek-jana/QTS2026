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
  Filter
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
  ScatterChart, 
  Scatter, 
  Label 
} from 'recharts';

// --- Components ---
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-full bg-slate-900 text-red-500 p-10 font-mono flex flex-col items-center justify-center">
          <ShieldAlert className="w-16 h-16 mb-4 animate-pulse" />
          <h1 className="text-2xl font-bold uppercase">Interface Override</h1>
          <button onClick={() => window.location.reload()} className="mt-8 bg-red-600 text-white px-12 py-3 font-black uppercase text-xs hover:bg-red-500">Reboot Cockpit</button>
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
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      width: chartContainerRef.current.clientWidth || 400,
      height: 250,
      timeScale: { borderColor: '#1e293b', timeVisible: true },
    });
    const series = chart.addSeries(CandlestickSeries, { upColor: '#10b981', downColor: '#ef4444' });
    chartRef.current = chart; seriesRef.current = series;
    return () => chart.remove();
  }, []);

  useEffect(() => {
    if (seriesRef.current && data && data.length > 0) {
      const seen = new Set();
      const cleanData = data.filter(d => { if (seen.has(d.time)) return false; seen.add(d.time); return true; }).sort((a, b) => a.time - b.time);
      seriesRef.current.setData(cleanData);
      chartRef.current.timeScale().fitContent();
    }
  }, [data]);

  return (
    <div className="w-full flex flex-col gap-1">
      <div className="text-[10px] font-mono text-emerald-500 uppercase font-bold tracking-tighter">{ticker} LIVE FEED</div>
      <div ref={chartContainerRef} className="w-full h-[250px] border border-slate-800 bg-black/50 shadow-inner" />
    </div>
  );
};

const Heatmap = ({ data, title }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!data || !data.length || !canvasRef.current) return;
    const canvas = canvasRef.current; const ctx = canvas.getContext('2d');
    const rows = data.length; const cols = data[0].length;
    canvas.width = cols; canvas.height = rows;
    const imageData = ctx.createImageData(cols, rows);
    const flatData = data.flat(); const maxVal = Math.max(...flatData.slice(0, 10000), 0.0001);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = data[i][j]; const ratio = Math.min(1.0, val / maxVal); const idx = (i * cols + j) * 4;
        imageData.data[idx] = Math.min(255, ratio * 450); imageData.data[idx + 1] = Math.max(0, (ratio - 0.3) * 500); 
        imageData.data[idx + 2] = Math.max(0, (ratio - 0.6) * 600); imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [data]);
  return (
    <div className="w-full h-full flex flex-col min-h-0">
      <div className="text-[10px] font-mono text-slate-500 mb-1 truncate font-bold uppercase">{title}</div>
      <div className="flex-1 min-h-0 relative border border-slate-800 bg-black overflow-hidden shadow-inner">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ imageRendering: 'pixelated' }} />
      </div>
    </div>
  );
};

const RankingGrid = ({ ladder, onSelectTicker, filterSector, onClearFilter }) => {
  const filtered = React.useMemo(() => {
    if (!ladder) return [];
    let items = ladder;
    if (filterSector) items = items.filter(it => it.sector === filterSector);
    const map = {};
    items.forEach(item => {
      if (!map[item.ticker] || item.score > map[item.ticker].score) {
        map[item.ticker] = { score: item.score, price: item.live_price, sector: item.sector };
      }
    });
    return Object.entries(map).map(([ticker, d]) => ({ ticker, ...d })).sort((a, b) => b.score - a.score);
  }, [ladder, filterSector]);

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar pr-2 h-[400px]">
      <div className="flex justify-between items-center mb-2 border-b border-emerald-900/30 pb-1">
        <div className="text-[10px] font-bold text-emerald-500/80 uppercase tracking-tighter">Decile Ladder</div>
        {filterSector && (
          <button onClick={onClearFilter} className="flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded-sm hover:bg-emerald-500/20 transition-all">
            <span className="text-[8px] font-bold text-emerald-400 uppercase tracking-widest">{filterSector}</span>
            <X className="w-2 h-2 text-emerald-400" />
          </button>
        )}
      </div>
      <table className="w-full text-[10px] border-collapse font-mono">
        <thead className="sticky top-0 bg-slate-900 shadow-sm text-slate-500 uppercase">
          <tr><th className="text-left py-2 font-normal">Ticker</th><th className="text-right py-2 pr-4 font-normal">Live Price</th><th className="text-right py-2 pr-4 font-normal">Z-Score</th><th className="text-right py-2 font-normal">Action</th></tr>
        </thead>
        <tbody>
          {filtered.map((row) => (
            <tr key={row.ticker} onClick={() => onSelectTicker(row.ticker)} className="border-b border-slate-800/40 hover:bg-emerald-500/10 cursor-pointer group transition-all">
              <td className="py-2.5 flex items-center gap-2 transition-all group-hover:translate-x-1"><div className={"w-1 h-3 " + (row.score > 0 ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-red-500")} /><span className="font-bold text-slate-200 group-hover:text-emerald-400">{row.ticker}</span></td>
              <td className="text-right py-2 text-slate-400 pr-4">${(row.price || 0).toFixed(2)}</td>
              <td className={"text-right py-2 pr-4 " + (row.score > 0 ? "text-emerald-400" : "text-red-400")}>{(row.score || 0).toFixed(4)}</td>
              <td className="text-right py-2"><span className={"text-[8px] px-1 border " + (row.score > 0 ? "border-emerald-800 text-emerald-600" : "border-red-900 text-red-700") + " uppercase font-black"}>{row.score > 0 ? "Long" : "Short"}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const Panel = ({ title, icon: Icon, children, className = "" }) => (
  <div className={"bg-slate-900 border border-slate-700 p-4 flex flex-col h-fit shadow-lg " + className}>
    <div className="flex-none flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
      <Icon className="w-4 h-4 text-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]" />
      <h2 className="text-xs font-mono uppercase tracking-[0.2em] text-slate-300 font-bold">{title}</h2>
    </div>
    <div className="flex-1">{children}</div>
  </div>
);

const MissionManual = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  const sections = [
    { title: "1. Spectral & Signal Viewer", items: [{ label: "Price Chart", detail: "Raw market fluid. GREEN = UP, RED = DOWN." }, { label: "Wavelet Spectrogram", detail: "Y-axis = Frequency. ENERGY = Brightness." }] },
    { title: "2. Statistical Integrity", items: [{ label: "ADF p-value", detail: "Numerical stability. MUST be < 0.05." }] },
    { title: "3. Feature Importance (SHAP)", items: [{ label: "SHAP Fusion", detail: "Bridges Modalities to Human Factors." }] },
    { title: "4. Execution Muscle", items: [{ label: "IS (BPS)", detail: "Execution efficiency." }, { label: "OMS Queue", detail: "Live execution pipe status." }] },
    { title: "5. Metacognition", items: [{ label: "Bayesian Belief", detail: "Model confidence score." }, { label: "Manifold Drift", detail: "t-SNE distance." }] },
    { title: "6. Sector Intelligence", items: [{ label: "Advanced Features", detail: "Filter ladder by sector." }] }
  ];
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 font-mono">
      <div className="bg-slate-900 border border-emerald-500/50 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
        <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-emerald-500/5"><div className="flex items-center gap-2 text-emerald-400 font-bold uppercase"><Terminal className="w-4 h-4" /> UQTS-2026 MANUAL</div><button onClick={onClose} className="text-slate-500 hover:text-white transition-colors bg-slate-800/50 p-1 rounded-full"><X className="w-4 h-4" /></button></div>
        <div className="p-8 space-y-8 overflow-y-auto no-scrollbar">
          {sections.map((s, i) => (
            <div key={i} className="space-y-4">
              <h3 className="text-emerald-500 font-bold uppercase text-[10px] tracking-widest border-b border-emerald-900/30 pb-2">{s.title}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">{s.items.map((it, j) => (<div key={j} className="bg-black/40 p-4 border border-slate-800/50 group hover:border-emerald-500/30 transition-all"><h4 className="text-slate-100 font-bold text-[9px] uppercase mb-1 flex items-center gap-2"><div className="w-1 h-2 bg-emerald-500" />{it.label}</h4><p className="text-slate-500 text-[10px] leading-relaxed italic">{it.detail}</p></div>))}</div>
            </div>
          ))}
        </div>
        <div className="p-4 bg-black/40 border-t border-slate-800 flex justify-center"><button onClick={onClose} className="bg-emerald-600 text-black px-12 py-3 font-black uppercase text-[10px] hover:bg-emerald-400 shadow-lg">ACKNOWLEDGE</button></div>
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
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/cockpit');
    ws.current.onopen = () => setStatus('active');
    ws.current.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.type === 'GLOBAL_UPDATE') setGlobalData(payload);
        else if (payload.type === 'SPECTRAL_UPDATE') setSpectralData(payload.spectral);
      } catch (err) {}
    };
    ws.current.onclose = () => setStatus('disconnected');
    return () => ws.current.close();
  }, []);

  const handleSelectTicker = (ticker) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ command: 'SET_TICKER', ticker }));
  };

  if (!globalData) return <div className="h-screen w-full bg-black flex flex-col items-center justify-center text-emerald-500 font-mono gap-4 animate-pulse"><Zap className="w-12 h-12" /> INITIALIZING ENGINE...</div>;

  const inst = globalData.institutional || {};
  const meta = globalData.metacognition || {};
  const exec = globalData.execution || {};

  return (
    <ErrorBoundary>
      <MissionManual isOpen={showManual} onClose={() => setShowManual(false)} />
      <div className="min-h-screen w-full bg-black text-slate-300 font-mono text-[11px] flex flex-col p-2 gap-2 overflow-x-hidden selection:bg-emerald-500/30">
        
        {/* Header */}
        <div className="flex-none h-16 flex justify-between items-center border-b border-slate-800 pb-2 px-2">
          <div className="flex items-center gap-4">
            <div className="text-2xl font-black text-emerald-500 tracking-tighter italic">UQTS-2026</div>
            <div className="flex items-center gap-2 border-l border-slate-800 pl-4"><div className={"w-2 h-2 rounded-full " + (status === "active" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" : "bg-red-500")} /><span className="uppercase text-[9px] font-bold text-slate-500">{status}</span></div>
            <div className="text-[9px] text-slate-600 border border-slate-800 px-2 py-0.5 rounded-sm uppercase tracking-tighter">SIM_TIME: {globalData.timestamp}</div>
            <div className="flex gap-6 px-6 border-l border-slate-800">
               <div className="flex flex-col"><span className="text-[8px] text-slate-600 uppercase font-black tracking-widest">Gross Exp</span><span className="text-xs font-bold text-slate-100 italic">{(inst.gross_exposure || 0).toFixed(1)}%</span></div>
               <div className="flex flex-col"><span className="text-[8px] text-slate-600 uppercase font-black tracking-widest">Net Exp</span><span className="text-xs font-bold text-slate-100">{(inst.net_exposure || 0).toFixed(1)}%</span></div>
               <div className="flex flex-col border-l border-slate-800 pl-6"><span className="text-[8px] text-slate-600 uppercase font-black tracking-widest">API Latency</span><span className="text-xs font-bold text-emerald-500 italic">{(inst.data_latency_ms || 0).toFixed(0)}ms</span></div>
            </div>
            <button onClick={() => setShowManual(true)} className="flex items-center gap-1.5 px-3 py-1 border border-emerald-900/30 hover:bg-emerald-500/10 transition-all text-slate-500 hover:text-emerald-400 group ml-4 outline-none"><HelpCircle className="w-3 h-3" /><span className="text-[9px] uppercase font-bold tracking-[0.2em]">Manual</span></button>
          </div>
          <button onClick={() => ws.current.send('KILL_SWITCH')} className="px-6 py-2 transition-all font-black uppercase text-[10px] border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-white shadow-[0_0_15px_rgba(220,38,38,0.2)]">Kill System</button>
        </div>

        <div className="flex-1 grid grid-cols-12 gap-2 pb-4 overflow-hidden">
          
          <Panel title="Spectral Alpha" icon={Activity} className="col-span-4 h-full overflow-y-auto no-scrollbar">
            <div className="flex flex-col gap-6">
              {spectralData ? (<><PriceChart data={spectralData.history} ticker={spectralData.ticker} /><div className="h-72 relative border border-slate-800 bg-black shadow-2xl"><Heatmap data={spectralData.cwt} title={spectralData.ticker + " Wavelet Spectrogram"} /></div></>) : (<div className="h-[330px] w-full flex items-center justify-center text-slate-700 italic text-[10px] uppercase tracking-widest animate-pulse border border-slate-900 bg-black/40">Select ticker from ladder to hydrate manifold...</div>)}
              <div className="grid grid-cols-2 gap-2"><div className="bg-slate-800/30 p-6 border border-slate-700 text-center"><div className="text-[8px] text-slate-500 uppercase font-black mb-1 tracking-widest">ADF p-value</div><div className={"text-2xl font-black " + (spectralData?.adf_p_value < 0.05 ? "text-emerald-500" : "text-slate-500")}>{(spectralData?.adf_p_value || 0.0001).toFixed(6)}</div></div><div className="bg-slate-800/30 p-6 border border-slate-700 text-center flex flex-col items-center justify-center"><div className="text-[8px] text-slate-500 uppercase font-black mb-1 tracking-widest">Alpha State</div><span className="text-emerald-400 text-[9px] font-black border border-emerald-900/50 px-2 py-0.5 rounded-sm shadow-md">READY</span></div></div>
              <div className="h-48 border-t border-slate-800 pt-4"><div className="text-[8px] text-slate-600 mb-2 uppercase font-black tracking-widest text-right">Idiosyncratic SHAP Fusion</div>{spectralData?.shap_values ? (<ResponsiveContainer width="100%" height="100%"><BarChart data={Object.entries(spectralData.shap_values).map(([name, val]) => ({ name, val }))} layout="vertical"><XAxis type="number" hide domain={[0, 1]} /><YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 8, fill: '#475569' }} /><Bar dataKey="val" fill="#10b981" radius={[0, 2, 2, 0]} /></BarChart></ResponsiveContainer>) : <div className="h-full w-full flex items-center justify-center text-slate-800 italic uppercase text-[8px] tracking-widest">Awaiting MODALITY_GATING...</div>}</div>
            </div>
          </Panel>

          <Panel title="Metacognition" icon={ShieldAlert} className="col-span-4 h-full overflow-y-auto no-scrollbar">
            <div className="flex flex-col gap-8">
               <div className="flex justify-between items-center bg-slate-800/30 p-8 border border-slate-700 shadow-2xl"><div className="flex flex-col"><span className="text-[10px] text-slate-600 uppercase font-black tracking-widest">Bayesian Belief</span><span className="text-6xl font-black text-emerald-400 drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]">{((meta.belief_score || 0) * 100).toFixed(2) + "%"}</span></div><Gauge className="w-16 h-16 text-emerald-500 opacity-20" /></div>
               <div className="h-72 border border-slate-800 bg-black/40 p-4"><div className="text-[8px] text-slate-600 mb-4 uppercase font-black text-center tracking-[0.4em]">Manifold Drift (t-SNE Latent)</div><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ bottom: 30, left: 10, right: 10 }}><CartesianGrid strokeDasharray="1 4" stroke="#1e293b" /><XAxis type="number" dataKey="x" stroke="#475569" fontSize={8}><Label value="Latent Dim 1" offset={-15} position="insideBottom" fill="#475569" fontSize={9} /></XAxis><YAxis type="number" dataKey="y" stroke="#475569" fontSize={8}><Label value="Latent Dim 2" angle={-90} position="insideLeft" fill="#475569" fontSize={9} /></YAxis><Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '9px'}} /><Scatter name="Base" data={(meta.manifold_drift || []).slice(0, 10).map(p => ({ x: p[0], y: p[1] }))} fill="#334155" shape="circle" /><Scatter name="Live" data={(meta.manifold_drift || []).slice(10).map(p => ({ x: p[0], y: p[1] }))} fill="#10b981" shape="cross" strokeWidth={3} /></ScatterChart></ResponsiveContainer></div>
               <div className="h-56 mt-4 border-t border-slate-800 pt-4"><div className="text-[8px] text-slate-600 mb-2 uppercase font-black tracking-widest">Alpha Decay Signal</div><ResponsiveContainer width="100%" height="100%"><LineChart data={(meta.alpha_decay || []).map((val, i) => ({ i, val }))} margin={{ bottom: 30, left: 10, right: 10 }}><XAxis dataKey="i" stroke="#475569" fontSize={8}><Label value="Events" offset={-15} position="insideBottom" fill="#475569" fontSize={9} /></XAxis><YAxis stroke="#475569" fontSize={8}><Label value="Gain" angle={-90} position="insideLeft" fill="#475569" fontSize={9} /></YAxis><Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '9px'}} itemStyle={{color: '#10b981'}} /><Line type="monotone" dataKey="val" stroke="#10b981" dot={{ r: 2, fill: '#10b981' }} activeDot={{ r: 5 }} strokeWidth={4} /></LineChart></ResponsiveContainer></div>
            </div>
          </Panel>

          <Panel title="Ranking Ladder" icon={TrendingUp} className="col-span-4 h-full overflow-hidden flex flex-col">
             <RankingGrid ladder={globalData.rankings?.ladder} onSelectTicker={handleSelectTicker} filterSector={selectedSector} onClearFilter={() => setSelectedSector(null)} />
             
             <div className="h-64 border-t border-slate-800 mt-4 pt-4 overflow-y-auto no-scrollbar">
                <div className="text-[8px] text-slate-500 mb-2 uppercase font-black tracking-widest flex justify-between items-center"><span>Sector Matrix (Interactive)</span><Filter className="w-3 h-3 text-slate-700" /></div>
                <div className="grid grid-cols-3 gap-1.5">
                  {Object.entries(inst.sector_exposure || {}).map(([sector, stats]) => (
                     <button 
                        key={sector} 
                        onClick={() => setSelectedSector(selectedSector === sector ? null : sector)}
                        className={"border p-2 flex flex-col justify-center transition-all text-left " + (selectedSector === sector ? "bg-emerald-500/20 border-emerald-500" : "bg-slate-800/30 border-slate-800 hover:bg-slate-800/50 hover:border-slate-600")}
                     >
                        <div className="text-[7px] text-slate-500 uppercase truncate font-black">{sector}</div>
                        <div className={"text-[10px] font-black " + (stats.exposure > 0 ? "text-emerald-500" : "text-rose-500")}>{(stats.exposure > 0 ? "+" : "") + stats.exposure.toFixed(1)}%</div>
                        <div className="flex justify-between items-center mt-1 border-t border-slate-700/50 pt-1">
                            <span className="text-[6px] text-slate-600 uppercase font-black">{stats.count} Tickers</span>
                            <span className="text-[6px] text-emerald-600/60 italic font-black">α: {(stats.avg_score || 0).toFixed(2)}</span>
                        </div>
                     </button>
                  ))}
                </div>
             </div>
             
             <div className="h-24 border-t border-slate-800 mt-6 pt-2 flex flex-col gap-1.5 overflow-hidden">
                <div className="grid grid-cols-2 gap-2 h-full">
                  <div className="bg-slate-800/20 p-1.5 border border-slate-800 flex flex-col justify-center"><span className="text-[6px] text-slate-500 uppercase block font-black">Champ Sharpe</span><span className="text-xs font-black text-cyan-400">1.42</span></div>
                  <div className="bg-slate-800/20 p-1.5 border border-slate-800 flex flex-col justify-center"><span className="text-[6px] text-slate-500 uppercase block font-black">Chal Sharpe</span><span className="text-xs font-black text-emerald-400">2.36</span></div>
                </div>
                <div className="bg-black/40 p-1.5 border border-slate-800 text-[7px] text-emerald-500/60 uppercase font-black truncate animate-pulse tracking-widest text-center">SYSTEM_PIPELINE: V1_ACTIVE</div>
             </div>
          </Panel>

          <Panel title="Execution & Reality Check" icon={Cpu} className="col-span-12 mt-2 h-fit">
            <div className="grid grid-cols-12 gap-8 p-2">
              <div className="col-span-3 flex flex-col justify-center gap-2 border-r border-slate-800 pr-8">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-tighter leading-none mb-1">Implementation Shortfall</div>
                <div className="text-4xl font-black text-slate-100 italic">{(exec.implementation_shortfall || 0).toFixed(2)} <span className="text-xs text-slate-700 not-italic uppercase">BPS</span></div>
                <div className="text-[9px] text-slate-500 font-mono italic">{"IS_VAR_95: " + (exec.is_var || 0).toFixed(6)}</div>
              </div>
              <div className="col-span-4 border-r border-slate-800 pr-8 flex flex-col gap-2">
                <div className="text-[9px] text-slate-500 uppercase font-black flex justify-between items-center tracking-widest"><span>OMS Queue (Live)</span><div className="flex gap-2"><span className="text-emerald-500">{(inst.oms_queue?.filled || 0)}F</span><span className="text-amber-500">{(inst.oms_queue?.working || 0)}W</span></div></div>
                <div className="bg-black/50 p-2 h-24 overflow-y-auto no-scrollbar font-mono text-[9px] border border-slate-800 shadow-inner">{(inst.order_log || []).map((log, i) => (<div key={i} className="flex justify-between border-b border-slate-900 pb-1 mb-1 hover:bg-emerald-500/5 transition-colors"><span className="text-slate-600">{log.time}</span><span className={log.side === 'BUY' ? 'text-emerald-500 font-bold' : 'text-rose-500 font-bold'}>{log.side + " " + log.ticker}</span><span className="text-slate-400 font-bold italic">{(log.status || "PENDING")}</span></div>))}</div>
              </div>
              <div className="col-span-5 flex justify-between items-center px-4">
                 <div className="flex flex-col gap-2 flex-1 max-w-[200px]"><div className="text-[9px] text-slate-600 uppercase font-black tracking-[0.2em] text-center">Slippage Heatmap</div><div className="grid grid-cols-5 gap-1 bg-black/40 p-2 border border-slate-800 shadow-xl">{(exec.slippage_heatmap || []).flat().map((v, i) => (<div key={i} className="w-3.5 h-3.5 bg-emerald-500 rounded-sm transition-all duration-500" style={{ opacity: v }} />))}</div></div>
                 <div className="flex flex-col items-end text-right gap-1 pt-4">
                    <div className="text-[10px] text-slate-600 italic uppercase font-black tracking-tighter italic">Muscle: C++26</div>
                    <div className="text-emerald-500 font-black text-[10px] shadow-[0_0_15px_rgba(16,185,129,0.2)] border border-emerald-900/50 px-3 py-1 uppercase tracking-widest bg-emerald-950/20">Linked_Ready</div>
                    <div className="text-[10px] text-slate-400 font-mono mt-1 uppercase font-bold tracking-widest">Latency: 84.2μs</div>
                    <div className="text-[8px] text-slate-800 font-bold uppercase">Direct OSQP Context</div>
                 </div>
              </div>
            </div>
          </Panel>

        </div>
      </div>
    </ErrorBoundary>
  );
}
