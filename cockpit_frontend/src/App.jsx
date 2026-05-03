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
  if (!data) return null;
  return (
    <div className="w-full h-full flex flex-col">
      <div className="text-[10px] font-mono text-slate-500 mb-1">{title}</div>
      <div className="flex-1 grid grid-cols-64 gap-[1px]">
        {data.map((row, i) => (
          row.map((val, j) => {
            const intensity = Math.min(255, Math.floor(val * 255));
            return (
              <div 
                key={`${i}-${j}`} 
                className="w-full h-[2px]" 
                style={{ backgroundColor: `rgb(${intensity}, ${intensity/2}, ${255-intensity})` }}
              />
            );
          })
        ))}
      </div>
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
      const payload = json.parse(event.data);
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
            <div className="flex-1 overflow-y-auto no-scrollbar pr-2">
              <div className="text-[10px] text-slate-500 mb-2 uppercase">Decile Ladder</div>
              <table className="w-full text-[10px]">
                <thead className="sticky top-0 bg-slate-900 border-b border-slate-800">
                  <tr>
                    <th className="text-left py-1 text-slate-500 uppercase">Ticker</th>
                    <th className="text-right py-1 text-slate-500 uppercase">Z-Score</th>
                    <th className="text-right py-1 text-slate-500 uppercase">Style</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rankings.ladder.map((row, i) => (
                    <tr key={row.ticker} className="border-b border-slate-800/50 group hover:bg-slate-800/30">
                      <td className="py-2 flex items-center gap-2">
                        <div className={`w-1 h-4 ${row.score > 0 ? 'bg-emerald-500' : 'bg-red-500'}`} />
                        <span className="font-bold text-slate-200">{row.ticker}</span>
                      </td>
                      <td className={`text-right py-2 ${row.score > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {row.score.toFixed(4)}
                      </td>
                      <td className="text-right text-slate-600">ALPHA_V1</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
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
               <span className="animate-pulse mr-2">></span> {data.pipeline.training_progress}
            </div>
          </div>
        </Panel>

      </div>
    </div>
  );
}
