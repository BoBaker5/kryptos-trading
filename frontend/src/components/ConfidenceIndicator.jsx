// components/ConfidenceIndicator.jsx
import React from 'react';
import { AlertCircle, BrainCircuit, ChartBar, Cpu } from 'lucide-react';

const ConfidenceIndicator = ({ signals }) => {
  const mockSignals = [
    {
      symbol: 'SOLUSD',
      ml: 0.470,
      ai: 0.492,
      tech: 0.524,
      final: 0.500
    },
    {
      symbol: 'AVAXUSD',
      ml: 0.470,
      ai: 0.496,
      tech: 0.502,
      final: 0.492
    },
    {
      symbol: 'XRPUSD',
      ml: 0.470,
      ai: 0.494,
      tech: 0.530,
      final: 0.502
    },
    {
      symbol: 'XDGUSD',
      ml: 0.470,
      ai: 0.497,
      tech: 0.525,
      final: 0.502
    },
    {
      symbol: 'SHIBUSD',
      ml: 0.470,
      ai: 0.490,
      tech: 0.502,
      final: 0.490
    }
  ];

  const getConfidenceColor = (score) => {
    if (score >= 0.52) return 'text-green-500';
    if (score <= 0.47) return 'text-red-500';
    return 'text-yellow-500';
  };

  const getConfidenceIcon = (score) => {
    if (score >= 0.52) return '🟢';
    if (score <= 0.47) return '🔴';
    return '⚪';
  };

  return (
    <div className="bg-[#001F3F] rounded-lg p-6">
      <h2 className="text-xl font-bold text-[#87CEEB] mb-4">Trading Signals</h2>
      <div className="space-y-4">
        {mockSignals.map((signal, index) => (
          <div key={index} className="bg-[#002851] p-4 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[#87CEEB] font-semibold">{signal.symbol}</span>
              <span className={`text-lg ${getConfidenceColor(signal.final)}`}>
                {getConfidenceIcon(signal.final)} {(signal.final * 100).toFixed(1)}%
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="flex items-center">
                <BrainCircuit className="h-4 w-4 mr-2 text-purple-400" />
                <span className="text-sm text-gray-400">ML: {(signal.ml * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center">
                <Cpu className="h-4 w-4 mr-2 text-blue-400" />
                <span className="text-sm text-gray-400">AI: {(signal.ai * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center">
                <ChartBar className="h-4 w-4 mr-2 text-green-400" />
                <span className="text-sm text-gray-400">Tech: {(signal.tech * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConfidenceIndicator;
