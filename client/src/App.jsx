import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, Activity, Mic, ShieldCheck, AlertTriangle, FileAudio, Server, RefreshCw } from 'lucide-react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const API_URL = "http://localhost:5003/api/analyze-audio";
const WS_URL = "ws://localhost:5003/ws/live-stream";

export default function App() {
  const [score, setScore] = useState(0);
  const [status, setStatus] = useState("Safe");
  const [logs, setLogs] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [waveform, setWaveform] = useState(Array(50).fill(0));
  const [metadata, setMetadata] = useState(null);
  const [isFileMode, setIsFileMode] = useState(false);

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const idleAnimationRef = useRef(null);

  useEffect(() => {
    let phase = 0;
    const animateIdle = () => {
      if (!isRecording && !isFileMode) {
        phase += 0.1;
        const idleWave = Array.from({ length: 50 }, (_, i) => {
          return 5 + Math.sin(phase + i * 0.2) * 5 + Math.random() * 2;
        });
        setWaveform(idleWave);
      }
      idleAnimationRef.current = requestAnimationFrame(animateIdle);
    };
    idleAnimationRef.current = requestAnimationFrame(animateIdle);
    
    return () => cancelAnimationFrame(idleAnimationRef.current);
  }, [isRecording, isFileMode]);

  const addLog = (msg) => {
    setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg }].slice(-6));
  };

  const updateScore = (newScore) => {
    setScore(newScore);
    if (newScore > 85) {
      setStatus("CRITICAL");
    } else if (newScore > 65) {
      setStatus("Soft Warning");
    } else {
      setStatus("Safe");
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    addLog(`Initiating intercept: ${file.name}`);
    setIsFileMode(true); // Stop idle animation
    
    // Extract real waveform for visualizer
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      
      const chunkSize = Math.floor(channelData.length / 50);
      const visData = [];
      for(let i=0; i<50; i++) {
        let maxVal = 0;
        const offset = i * chunkSize;
        for(let j=0; j<chunkSize; j++) {
           if (offset + j < channelData.length) {
               const val = Math.abs(channelData[offset + j]);
               if (val > maxVal) maxVal = val;
           }
        }
        visData.push(maxVal * 200); // Scale up for visualizer
      }
      setWaveform(visData);
    } catch (e) {
      console.error("Audio visualization error:", e);
    }
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      updateScore(res.data.authenticity_score);
      setMetadata(res.data.metadata);
      addLog(`Analysis complete: ${res.data.classification}`);
    } catch (error) {
      addLog(`Upload error: ${error.message}`);
    }
  };

  // simulateWaveform is no longer needed since we use real data
  const simulateWaveform = () => {};

  const startLiveStream = async () => {
    try {
      setIsFileMode(false); // Re-enable mic/idle mode
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      
      const audioContext = audioContextRef.current;
      await audioContext.audioWorklet.addModule(
        URL.createObjectURL(new Blob([`
          class Processor extends AudioWorkletProcessor {
            process(inputs) {
              const input = inputs[0];
              if (input.length > 0) {
                this.port.postMessage(input[0]);
              }
              return true;
            }
          }
          registerProcessor('processor', Processor);
        `], { type: 'application/javascript' }))
      );

      wsRef.current = new WebSocket(WS_URL);
      
      wsRef.current.onopen = () => {
        addLog("Secure WebSocket Uplink Established");
        setIsRecording(true);
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateScore(data.authenticity_score);
        setMetadata(data.metadata);
        addLog(`Inference: ${data.classification} (${data.authenticity_score.toFixed(1)}%)`);
      };

      const source = audioContext.createMediaStreamSource(mediaStreamRef.current);
      workletNodeRef.current = new AudioWorkletNode(audioContext, 'processor');
      
      const envelopeData = Array(50).fill(0);

      workletNodeRef.current.port.onmessage = (e) => {
        const inputData = e.data;
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const pcmData = new Int16Array(inputData.length);
          let maxVal = 0;
          for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
            maxVal = Math.max(maxVal, Math.abs(inputData[i]));
          }
          wsRef.current.send(pcmData.buffer);

          envelopeData.shift();
          envelopeData.push(maxVal * 400);

          setWaveform([...envelopeData]);
        }
      };

      source.connect(workletNodeRef.current);
      workletNodeRef.current.connect(audioContext.destination);

    } catch (err) {
      addLog(`Mic Access Denied: ${err.message}`);
    }
  };

  const stopLiveStream = () => {
    if (wsRef.current) wsRef.current.close();
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(t => t.stop());
    if (audioContextRef.current) audioContextRef.current.close();
    setIsRecording(false);
    addLog("Microphone stream severed.");
  };

  const toggleRecording = () => {
    isRecording ? stopLiveStream() : startLiveStream();
  };

  const isSynthetic = status === 'CRITICAL' || status === 'Soft Warning';
  
  const chartData = {
    labels: Array(50).fill(''),
    datasets: [{
      label: 'Acoustic Energy',
      data: waveform,
      borderColor: isSynthetic ? '#f43f5e' : '#34d399',
      backgroundColor: isSynthetic ? 'rgba(244, 63, 94, 0.2)' : 'rgba(52, 211, 153, 0.2)',
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    scales: { 
      y: { display: false, min: 0, max: 100 }, 
      x: { display: false } 
    },
    plugins: { legend: { display: false }, tooltip: { enabled: false } }
  };

  return (
    <div className="min-h-screen bg-[#050510] text-gray-200 p-6 font-sans overflow-x-hidden relative">
      <div className="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-blue-600/20 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[30vw] h-[30vw] bg-purple-600/20 rounded-full blur-[100px] pointer-events-none"></div>

      <header className="flex justify-between items-center mb-10 border-b border-white/5 pb-6 relative z-10">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-linear-to-br from-blue-500/20 to-purple-500/20 rounded-xl border border-blue-400/30 shadow-[0_0_20px_rgba(59,130,246,0.3)]">
            <Activity className="w-8 h-8 text-blue-400 animate-pulse" />
          </div>
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-blue-400 to-purple-400">
              Authentix
            </h1>
            <p className="text-gray-400 text-sm font-medium tracking-wide">Next-Gen Audio Forensics Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-gray-400 bg-white/5 backdrop-blur-xl px-5 py-2.5 rounded-full border border-white/10 shadow-inner">
          <Server className="w-4 h-4" /> Nexus Link: 
          <span className="flex items-center gap-2 text-emerald-400 font-semibold shadow-[0_0_10px_rgba(52,211,153,0.5)]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping absolute"></span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 relative"></span>
            {isRecording ? "STREAMING" : "ONLINE"}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 relative z-10 max-w-7xl mx-auto">
        <div className="space-y-8 flex flex-col">
          <div className="group bg-white/2 backdrop-blur-md rounded-2xl p-8 flex flex-col items-center justify-center border border-white/10 hover:border-blue-500/50 hover:bg-blue-500/5 transition-all duration-300 relative overflow-hidden shadow-lg">
            <div className="absolute inset-0 bg-linear-to-b from-blue-500/0 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <input type="file" className="absolute inset-0 opacity-0 cursor-pointer z-10" accept="audio/*" onChange={handleFileUpload} />
            <UploadCloud className="w-16 h-16 text-blue-400 mb-4 group-hover:scale-110 transition-transform duration-300" />
            <h3 className="font-bold text-xl text-white tracking-wide">Inject Audio File</h3>
            <p className="text-sm text-gray-500 mt-2 text-center">WAV • MP3 • FLAC<br/>(Drag & Drop)</p>
          </div>

          <button 
            onClick={toggleRecording}
            className={`flex items-center justify-center gap-3 p-5 rounded-2xl font-bold text-lg transition-all duration-300 shadow-lg ${
              isRecording 
                ? 'bg-rose-500/10 border border-rose-500/50 text-rose-400 shadow-[0_0_30px_rgba(244,63,94,0.3)]' 
                : 'bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 border border-blue-400/30 text-white shadow-[0_0_20px_rgba(59,130,246,0.2)]'
            }`}
          >
            {isRecording ? <RefreshCw className="w-6 h-6 animate-spin" /> : <Mic className="w-6 h-6" />}
            {isRecording ? "Intercepting Live Audio..." : "Engage Live Microphone"}
          </button>

          <div className="bg-white/2 backdrop-blur-md rounded-2xl p-6 flex-1 border border-white/10 shadow-lg">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2 text-gray-300"><FileAudio className="w-5 h-5 text-purple-400" /> System Logs</h3>
            <div className="space-y-3 font-mono text-xs">
              {logs.map((l, i) => (
                <div key={i} className="flex gap-3 text-gray-400 bg-black/20 p-2 rounded-lg border border-white/5">
                  <span className="text-blue-400">[{l.time}]</span>
                  <span className={l.msg.includes("CRITICAL") ? "text-rose-400 font-bold" : "text-gray-200"}>{l.msg}</span>
                </div>
              ))}
              {logs.length === 0 && <div className="text-gray-500 italic p-2">Awaiting telemetry data...</div>}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-8 flex flex-col">
          <div className={`backdrop-blur-md rounded-2xl p-10 relative overflow-hidden flex flex-col items-center justify-center min-h-75 transition-all duration-700 shadow-2xl border ${
            status === 'CRITICAL' ? 'border-rose-500/50 bg-rose-950/20 shadow-[0_0_50px_rgba(244,63,94,0.15)]' : 
            status === 'Soft Warning' ? 'border-amber-500/50 bg-amber-950/20 shadow-[0_0_50px_rgba(245,158,11,0.15)]' : 
            'border-emerald-500/30 bg-emerald-950/10 shadow-[0_0_50px_rgba(52,211,153,0.1)]'
          }`}>
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiLz48L3N2Zz4=')] opacity-50"></div>
            
            <div className="text-center z-10">
              <h2 className="text-gray-400 text-sm uppercase tracking-[0.3em] font-bold mb-4">Deepfake Probability Score</h2>
              <div className="flex items-center justify-center gap-4 mb-6">
                <span className={`text-8xl font-black tracking-tighter drop-shadow-lg ${
                  status === 'CRITICAL' ? 'text-rose-500' : 
                  status === 'Soft Warning' ? 'text-amber-500' : 
                  'text-emerald-400'
                }`}>
                  {score.toFixed(1)}<span className="text-5xl">%</span>
                </span>
              </div>
              
              <div className={`inline-flex items-center gap-3 px-8 py-3 rounded-full font-extrabold text-sm uppercase tracking-widest backdrop-blur-xl ${
                  status === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/50 shadow-[0_0_20px_rgba(244,63,94,0.4)]' : 
                  status === 'Soft Warning' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50 shadow-[0_0_20px_rgba(245,158,11,0.4)]' : 
                  'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 shadow-[0_0_20px_rgba(52,211,153,0.4)]'
              }`}>
                {status === 'CRITICAL' && <AlertTriangle className="w-6 h-6 animate-pulse" />}
                {status === 'Safe' && <ShieldCheck className="w-6 h-6" />}
                {status === 'CRITICAL' ? 'SYNTHETIC DETECTED' : status === 'Safe' ? 'AUTHENTIC' : status}
              </div>
            </div>
          </div>

          <div className="bg-white/2 backdrop-blur-md rounded-2xl p-8 flex-1 flex flex-col border border-white/10 shadow-lg">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-semibold text-lg text-gray-200">Acoustic Signal Array</h3>
              <div className="text-xs font-mono text-blue-400 bg-blue-400/10 px-3 py-1 rounded-md border border-blue-400/20">LIVE</div>
            </div>
            <div className="flex-1 w-full min-h-55 relative bg-black/40 rounded-xl border border-white/5 p-4">
               <Line data={chartData} options={chartOptions} />
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 pt-6 border-t border-white/10 text-center">
               <div className="bg-white/5 p-3 rounded-lg border border-white/5 hover:bg-white/10 transition-colors">
                  <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">Architecture</div>
                  <div className="font-semibold text-blue-400">Hybrid ML+DSP</div>
               </div>
               <div className="bg-white/5 p-3 rounded-lg border border-white/5 hover:bg-white/10 transition-colors">
                  <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">Source Target</div>
                  <div className="font-semibold text-purple-400">{metadata?.source || "Standby"}</div>
               </div>
               <div className="bg-white/5 p-3 rounded-lg border border-white/5 hover:bg-white/10 transition-colors">
                  <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">HF Density Ratio</div>
                  <div className="font-semibold text-emerald-400">{metadata?.hf_ratio || "--"}</div>
               </div>
               <div className="bg-white/5 p-3 rounded-lg border border-white/5 hover:bg-white/10 transition-colors">
                  <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">Stream Bandwidth</div>
                  <div className="font-semibold text-amber-400">{metadata?.bandwidth || "-- Hz"}</div>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
