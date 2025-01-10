// components/PerformanceChart.jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PerformanceChart = ({ data }) => {
  // Mock data for demonstration
  const mockData = [
    { timestamp: '00:00', balance: 49.21 },
    { timestamp: '01:00', balance: 49.35 },
    { timestamp: '02:00', balance: 49.28 },
    { timestamp: '03:00', balance: 49.45 },
    { timestamp: '04:00', balance: 49.30 },
    { timestamp: '05:00', balance: 49.20 },
  ];

  return (
    <div className="bg-[#001F3F] rounded-lg p-6">
      <h2 className="text-xl font-bold text-[#87CEEB] mb-4">Portfolio Performance</h2>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data || mockData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a8a" />
            <XAxis 
              dataKey="timestamp" 
              stroke="#94a3b8"
            />
            <YAxis 
              stroke="#94a3b8"
              domain={['auto', 'auto']}
              tickFormatter={(value) => `$${value.toFixed(2)}`}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#002851', 
                border: '1px solid #1e3a8a',
                color: '#87CEEB'
              }}
              formatter={(value) => [`$${value.toFixed(2)}`, 'Balance']}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="balance" 
              stroke="#87CEEB" 
              dot={false}
              name="Portfolio Value"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceChart;
