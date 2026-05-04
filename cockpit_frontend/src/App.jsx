import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  ShieldAlert, 
  TrendingUp, 
  Zap, 
  Terminal, 
  Cpu, 
  Gauge, 
  AlertTriangle 
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
    if (!data || !canvasRef.current) return;
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
                ${row.price > 0 ? row.price.toFixed(2) : "---"}
              </td>
              <td className={`text-right py-2.5 font-mono ${row.score > 0 ? 'text-emerald-400' : 'text-red-400'} pr-4`}>
                {row.score.toFixed(4)}
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

export default function MissionControl() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState('connecting');
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/cockpit');
    
    ws.current.onopen = () => setStatus('active');
    ws.current.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setData(payload);
    };
    ws.current.onclose = () => setStatus('disconnected');

    return () => ws.current.close();
  }, []);

  const triggerKillSwitch = () => {
    if (ws.current) ws.current.send('KILL_SWITCH');
  };

  const handleSelectTicker = (ticker) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: 'SET_TICKER', ticker }));
    }
  };

  if (!data) return (
    <div className="h-screen w-full bg-black flex items-center justify-center text-emerald-500 font-mono">
      <Zap className="animate-pulse mr-2" /> INITIALIZING COCKPIT SYSTEM...
    </div>
  );

  return (
    <div className="min-h-screen w-full bg-black text-slate-300 font-mono text-xs flex flex-col p-2 gap-2">
      {/* Header */}
      <div className="flex-none h-16 flex justify-between items-center border-b border-slate-800 pb-2">
        <div className="flex items-center gap-4">
          <div className="text-lg font-bold text-emerald-500">UQTS-2026 MISSION CONTROL</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-red-500'}`} />
            <span className="uppercase text-[10px]">{status}</span>
          </div>
          <div className="text-[10px] text-slate-500 font-mono">KNOWLEDGE_TIME: {data.timestamp}</div>
        </div>
        <button 
          onClick={triggerKillSwitch}
          className="bg-red-900/30 border border-red-500 text-red-500 px-6 py-2 hover:bg-red-500 hover:text-white transition-all font-bold uppercase tracking-tighter"
        >
          Emergency Kill Switch
        </button>
      </div>

      {/* Main Container - No longer fixed height */}
      <div className="flex-1 grid grid-cols-12 gap-2 pb-8">
        
        {/* 1. Spectral & Signal Viewer */}
        <Panel title="Spectral & Signal Viewer" icon={Activity} className="col-span-4 h-fit">
          <div className="flex flex-col gap-6">
            <div className="h-72 relative border border-slate-800 bg-black shadow-inner">
              <Heatmap data={data.spectral.cwt} title={`${data.spectral.ticker} Wavelet Spectrogram (Morlet)`} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-800/50 p-6 border border-slate-700 flex flex-col justify-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-tighter leading-none mb-1">ADF p-value</div>
                <div className={`text-3xl font-bold ${data.spectral.adf_p_value < 0.05 ? 'text-emerald-500' : 'text-red-500 animate-pulse'}`}>
                  {data.spectral.adf_p_value.toFixed(6)}
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
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={Object.entries(data.spectral.shap_values).map(([name, val]) => ({ name, val }))} layout="vertical">
                  <XAxis type="number" hide domain={[0, 1]} />
                  <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Bar dataKey="val" fill="#10b981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>

        {/* 2. Metacognition Panel */}
        <Panel title="Metacognition Panel" icon={ShieldAlert} className="col-span-4 h-fit">
          <div className="flex flex-col gap-8">
             <div className="flex justify-between items-center bg-slate-800/50 p-8 border border-slate-700 shadow-md">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Bayesian Belief Score</span>
                  <span className="text-5xl font-bold text-emerald-400">{(data.metacognition.belief_score * 100).toFixed(2)}%</span>
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
                    <Scatter name="Training" data={data.metacognition.manifold_drift.slice(0, 5).map(p => ({ x: p[0], y: p[1] }))} fill="#475569" shape="circle" />
                    <Scatter name="Live" data={data.metacognition.manifold_drift.slice(5).map(p => ({ x: p[0], y: p[1] }))} fill="#10b981" shape="cross" />
                  </ScatterChart>
                </ResponsiveContainer>
             </div>

             <div className="h-64 mt-4">
                <div className="text-[10px] text-slate-500 mb-2 uppercase tracking-widest">Alpha Decay (Cumulative Information Gain)</div>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.metacognition.alpha_decay.map((val, i) => ({ i, val }))}>
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
               <RankingGrid ladder={data.rankings.ladder} onSelectTicker={handleSelectTicker} />
            </div>
            
            <div className="h-72 border-t border-slate-800 pt-6 mt-4">
              <div className="text-[10px] text-slate-500 mb-2 uppercase font-bold tracking-widest">L/S Equity Spread (Cumulative Return %)</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={data.rankings.ls_spread.map((val, i) => ({ i, val }))}
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
          <div className="grid grid-cols-4 h-full gap-8 p-4">
            <div className="col-span-1 flex flex-col justify-center gap-4 border-r border-slate-800 pr-4">
              <div className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Implementation Shortfall</div>
              <div className="text-4xl font-bold text-slate-100">{data.execution.implementation_shortfall.toFixed(2)} <span className="text-sm text-slate-600 font-normal">BPS</span></div>
              {data.execution.needs_retune && (
                <div className="text-[10px] text-red-500 flex items-center gap-2 animate-pulse bg-red-900/20 p-2 border border-red-900">
                  <AlertTriangle className="w-3 h-3" /> RL Agent: RETUNE REQUIRED
                </div>
              )}
              <div className="text-[10px] text-slate-400 font-mono">VAR: {data.execution.is_var.toFixed(6)}</div>
            </div>
            <div className="col-span-2">
              <div className="text-[10px] text-slate-500 mb-4 uppercase text-center font-bold tracking-widest">LOB Slippage Heatmap (Energy Distribution)</div>
              <div className="grid grid-cols-5 grid-rows-5 h-40 gap-[3px] bg-slate-800/20 p-2 border border-slate-800 shadow-inner">
                {data.execution.slippage_heatmap.flat().map((v, i) => (
                   <div key={i} className="bg-emerald-500 transition-all duration-300" style={{ opacity: v }} />
                ))}
              </div>
            </div>
            <div className="col-span-1 flex flex-col justify-center items-end text-right gap-2">
              <div className="text-[11px] text-slate-500 uppercase italic font-bold tracking-tighter">Execution Muscle</div>
              <div className="text-[12px] font-bold text-emerald-500 bg-emerald-900/20 px-2 py-1 border border-emerald-900 shadow-[0_0_10px_rgba(16,185,129,0.2)]">C++26 [LINKED]</div>
              <div className="text-[10px] text-slate-400 font-mono">LATENCY: 84.2μs</div>
              <div className="text-[10px] text-slate-600">STABILITY: NOMINAL</div>
            </div>
          </div>
        </Panel>

        {/* 5. Pipeline Control */}
        <Panel title="Research Pipeline Control" icon={Terminal} className="col-span-4 h-fit">
          <div className="flex flex-col gap-6 h-full justify-center">
            <div className="flex justify-around items-center border-b border-slate-800 pb-4">
               <div className="flex flex-col items-center">
                  <span className="text-slate-500 uppercase text-[10px] font-bold">Champion</span>
                  <span className="text-emerald-500 text-xl font-bold underline cursor-help">SHARPE {data.pipeline.champion_sharpe}</span>
               </div>
               <div className="flex flex-col items-center">
                  <span className="text-slate-500 uppercase text-[10px] font-bold">Challenger</span>
                  <span className="text-cyan-400 text-xl font-bold underline cursor-help">SHARPE {data.pipeline.challenger_sharpe}</span>
               </div>
            </div>
            <div className="overflow-hidden bg-black p-4 border border-slate-800 text-emerald-500/80 font-mono text-[11px] shadow-inner h-24">
               <span className="animate-pulse mr-2 text-white font-bold">&gt;</span> {data.pipeline.training_progress}
            </div>
          </div>
        </Panel>

      </div>
    </div>
  );
}
