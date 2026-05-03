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
  <div className={`bg-slate-900 border border-slate-700 p-4 flex flex-col h-full ${className}`}>
    <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
      <Icon className="w-4 h-4 text-emerald-500" />
      <h2 className="text-xs font-mono uppercase tracking-widest text-slate-300 font-bold">{title}</h2>
    </div>
    <div className="flex-1 overflow-hidden">
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

const RankingGrid = ({ ladder }) => {
  // Fix 2: Unique-ify the list. One entry per ticker, highest score.
  const aggregated = React.useMemo(() => {
    const map = {};
    ladder.forEach(item => {
      if (!map[item.ticker] || item.score > map[item.ticker]) {
        map[item.ticker] = item.score;
      }
    });
    return Object.entries(map)
      .map(([ticker, score]) => ({ ticker, score }))
      .sort((a, b) => b.score - a.score);
  }, [ladder]);

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar pr-2">
      <div className="text-[10px] font-bold text-emerald-500/80 mb-2 uppercase border-b border-emerald-900/30 pb-1">Decile Ladder (House View)</div>
      <table className="w-full text-[10px] border-collapse">
        <thead className="sticky top-0 bg-slate-900 shadow-sm">
          <tr>
            <th className="text-left py-2 text-slate-500 uppercase font-normal">Ticker</th>
            <th className="text-right py-2 text-slate-500 uppercase font-normal pr-4">Z-Score</th>
            <th className="text-right py-2 text-slate-500 uppercase font-normal">Action</th>
          </tr>
        </thead>
        <tbody>
          {aggregated.map((row, i) => (
            <tr key={row.ticker} className="border-b border-slate-800/40 group hover:bg-emerald-500/5 transition-colors">
              <td className="py-2.5 flex items-center gap-2">
                <div className={`w-1 h-3 ${row.score > 0 ? 'bg-emerald-500' : 'bg-red-500'} shadow-sm`} />
                <span className="font-bold text-slate-200 tracking-tight">{row.ticker}</span>
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

  if (!data) return (
    <div className="h-screen w-full bg-black flex items-center justify-center text-emerald-500 font-mono">
      <Zap className="animate-pulse mr-2" /> INITIALIZING COCKPIT SYSTEM...
    </div>
  );

  return (
    <div className="h-screen w-full bg-black text-slate-300 font-mono text-xs overflow-hidden flex flex-col p-2 gap-2">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-slate-800 pb-2">
        <div className="flex items-center gap-4">
          <div className="text-lg font-bold text-emerald-500">UQTS-2026 MISSION CONTROL</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-red-500'}`} />
            <span className="uppercase text-[10px]">{status}</span>
          </div>
          <div className="text-[10px] text-slate-500">KNOWLEDGE_TIME: {data.timestamp}</div>
        </div>
        <button 
          onClick={triggerKillSwitch}
          className="bg-red-900/30 border border-red-500 text-red-500 px-4 py-1 hover:bg-red-500 hover:text-white transition-all font-bold uppercase"
        >
          Emergency Kill Switch
        </button>
      </div>

      {/* Grid Layout */}
      <div className="flex-1 grid grid-cols-12 grid-rows-6 gap-2">
        
        {/* 1. Spectral & Signal Viewer */}
        <Panel title="Spectral & Signal Viewer" icon={Activity} className="col-span-4 row-span-3">
          <div className="flex flex-col gap-4 h-full">
            <div className="h-1/2">
              <Heatmap data={data.spectral.cwt} title={`${data.spectral.ticker} Wavelet Spectrogram (Morlet)`} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-800/50 p-2 border border-slate-700">
                <div className="text-[10px] text-slate-500 uppercase">ADF p-value</div>
                <div className={`text-xl font-bold ${data.spectral.adf_p_value < 0.05 ? 'text-emerald-500' : 'text-red-500 animate-pulse'}`}>
                  {data.spectral.adf_p_value.toFixed(6)}
                </div>
                <div className="text-[8px] text-slate-600">STATIONARITY: {data.spectral.adf_p_value < 0.05 ? 'VERIFIED' : 'FAILED'}</div>
              </div>
              <div className="bg-slate-800/50 p-2 border border-slate-700">
                <div className="text-[10px] text-slate-500 uppercase">Active Modalities</div>
                <div className="flex gap-2 mt-1">
                  <span className="bg-emerald-900/50 text-emerald-400 px-1 border border-emerald-800 text-[8px]">LSTM</span>
                  <span className="bg-emerald-900/50 text-emerald-400 px-1 border border-emerald-800 text-[8px]">ViT</span>
                </div>
              </div>
            </div>
            <div className="flex-1 h-32 mt-2">
              <div className="text-[10px] text-slate-500 mb-2 uppercase">Feature SHAP Stream</div>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={Object.entries(data.spectral.shap_values).map(([name, val]) => ({ name, val }))} layout="vertical">
                  <XAxis type="number" hide domain={[0, 1]} />
                  <YAxis dataKey="name" type="category" width={70} tick={{ fontSize: 8, fill: '#94a3b8' }} />
                  <Bar dataKey="val" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>

        {/* 2. Metacognition Panel */}
        <Panel title="Metacognition Panel" icon={ShieldAlert} className="col-span-4 row-span-3">
          <div className="flex flex-col h-full gap-4">
             <div className="flex justify-between items-center bg-slate-800/50 p-4 border border-slate-700">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Bayesian Belief Score</span>
                  <span className="text-3xl font-bold text-emerald-400">{(data.metacognition.belief_score * 100).toFixed(2)}%</span>
                </div>
                <Gauge className="w-12 h-12 text-emerald-500 opacity-50" />
             </div>
             
             <div className="h-40">
                <div className="text-[10px] text-slate-500 mb-2 uppercase">Adversarial Drift (Training vs Live)</div>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <Scatter name="Training" data={data.metacognition.manifold_drift.slice(0, 5).map(p => ({ x: p[0], y: p[1] }))} fill="#475569" shape="circle" />
                    <Scatter name="Live" data={data.metacognition.manifold_drift.slice(5).map(p => ({ x: p[0], y: p[1] }))} fill="#10b981" shape="cross" />
                  </ScatterChart>
                </ResponsiveContainer>
             </div>

             <div className="flex-1">
                <div className="text-[10px] text-slate-500 mb-2 uppercase">Alpha Decay Curve (30D)</div>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.metacognition.alpha_decay.map((val, i) => ({ i, val }))}>
                    <Line type="monotone" dataKey="val" stroke="#10b981" dot={false} strokeWidth={2} />
                    <XAxis hide />
                    <YAxis hide />
                  </LineChart>
                </ResponsiveContainer>
             </div>
          </div>
        </Panel>

        {/* 3. Cross-Sectional Ranking Grid */}
        <Panel title="Ranking Grid" icon={TrendingUp} className="col-span-4 row-span-6">
          <div className="flex flex-col h-full">
            <RankingGrid ladder={data.rankings.ladder} />
            
            <div className="h-40 border-t border-slate-800 pt-4 mt-4">
              <div className="text-[10px] text-slate-500 mb-2 uppercase">L/S Equity Spread View</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.rankings.ls_spread.map((val, i) => ({ i, val }))}>
                  <Line type="stepAfter" dataKey="val" stroke="#10b981" dot={false} strokeWidth={2} />
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis hide />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>

        {/* 4. Execution & Reality Check */}
        <Panel title="Execution & Reality Check" icon={Cpu} className="col-span-8 row-span-2">
          <div className="grid grid-cols-4 h-full gap-4">
            <div className="col-span-1 flex flex-col justify-center gap-2">
              <div className="text-[10px] text-slate-500 uppercase">Implementation Shortfall</div>
              <div className="text-2xl font-bold text-slate-100">{data.execution.implementation_shortfall.toFixed(2)} <span className="text-xs text-slate-600 font-normal">BPS</span></div>
              <div className="text-[8px] text-red-500 flex items-center gap-1">
                <AlertTriangle className="w-2 h-2" /> RL Agent needs retune
              </div>
            </div>
            <div className="col-span-2">
              <div className="text-[10px] text-slate-500 mb-2 uppercase text-center">Slippage Heatmap (Limit Order Book)</div>
              <div className="grid grid-cols-5 grid-rows-5 h-20 gap-[2px]">
                {data.execution.slippage_heatmap.flat().map((v, i) => (
                   <div key={i} className="bg-emerald-500" style={{ opacity: v }} />
                ))}
              </div>
            </div>
            <div className="col-span-1 flex flex-col justify-center items-end text-right">
              <div className="text-[10px] text-slate-500 uppercase italic">Execution Muscle</div>
              <div className="text-[10px] font-bold text-emerald-500">C++26 [CONNECTED]</div>
              <div className="text-[8px] text-slate-600">Latency: 84μs</div>
            </div>
          </div>
        </Panel>

        {/* 5. Pipeline Control */}
        <Panel title="Research Pipeline Control" icon={Terminal} className="col-span-8 row-span-1">
          <div className="flex justify-between items-center h-full">
            <div className="flex gap-8 items-center">
               <div className="flex items-center gap-2">
                  <span className="text-slate-500 uppercase text-[10px]">Champion:</span>
                  <span className="text-emerald-500 font-bold underline cursor-help">Sharpe {data.pipeline.champion_sharpe}</span>
               </div>
               <div className="flex items-center gap-2">
                  <span className="text-slate-500 uppercase text-[10px]">Challenger:</span>
                  <span className="text-cyan-400 font-bold underline cursor-help">Sharpe {data.pipeline.challenger_sharpe}</span>
               </div>
            </div>
            <div className="flex-1 ml-10 overflow-hidden bg-black p-2 border border-slate-800 text-emerald-500/80 text-[10px]">
               <span className="animate-pulse mr-2">&gt;</span> {data.pipeline.training_progress}
            </div>
          </div>
        </Panel>

      </div>
    </div>
  );
}
