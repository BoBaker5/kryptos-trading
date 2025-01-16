import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, DollarSign, LineChart } from 'lucide-react';

const BotDashboard = ({ mode = 'live' }) => {
  const API_URL = process.env.REACT_APP_API_URL || '/api';
  
  const [botData, setBotData] = useState({
    status: 'stopped',
    positions: [],
    balance: {},
    metrics: {
      current_equity: mode === 'demo' ? 1000000 : 0,
      pnl: 0,
      pnl_percentage: 0
    },
    trades: []
  });

  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchBotStatus = async () => {
    try {
      const endpoint = `/api/bot-status/${mode}`;
      console.log('Fetching from:', `${API_URL}${endpoint}`);
      const response = await axios.get(`${API_URL}${endpoint}`);
      
      if (response.data.status === 'success') {
        setBotData(response.data.data);
        setError(null);
      }
    } catch (err) {
      console.error('Error fetching bot status:', err);
      setError('Unable to connect to trading server. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartBot = async () => {
    try {
      const endpoint = `/api/start-bot/${mode}`;
      const response = await axios.post(`${API_URL}${endpoint}`);
      if (response.data.status === 'success') {
        setBotData(prev => ({ ...prev, status: 'running' }));
      }
    } catch (err) {
      setError('Failed to start bot. Please try again.');
    }
  };

  const handleStopBot = async () => {
    try {
      const endpoint = `/api/stop-bot/${mode}`;
      const response = await axios.post(`${API_URL}${endpoint}`);
      if (response.data.status === 'success') {
        setBotData(prev => ({ ...prev, status: 'stopped' }));
      }
    } catch (err) {
      setError('Failed to stop bot. Please try again.');
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const endpoint = `/api/bot-status/${mode}`;
        console.log('Fetching from:', `${API_URL}${endpoint}`);
        const response = await axios.get(`${API_URL}${endpoint}`);
        
        if (response.data.status === 'success') {
          setBotData(response.data.data);
          setError(null);
        }
      } catch (err) {
        console.error('Error fetching bot status:', err);
        setError('Unable to connect to trading server. Please check your connection.');
      } finally {
        setIsLoading(false);
      }
    };
  
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [mode, API_URL]);

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-[#001F3F]">
          {mode === 'demo' ? 'Demo Trading Dashboard' : 'Live Trading Dashboard'}
        </h1>
        
        <div className="flex items-center gap-4">
          <div className={`px-3 py-1 rounded-full ${botData.status === 'running' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {botData.status === 'running' ? 'Active' : 'Stopped'}
          </div>
          
          {botData.status === 'running' ? (
            <button
              onClick={handleStopBot}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
            >
              Stop Bot
            </button>
          ) : (
            <button
              onClick={handleStartBot}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
            >
              Start Bot
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded">
          {error}
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500">Portfolio Value</p>
              <h3 className="text-2xl font-bold">
                ${botData.metrics?.current_equity?.toFixed(2) || '0.00'}
              </h3>
            </div>
            <DollarSign className="h-8 w-8 text-[#87CEEB]" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500">P&L</p>
              <h3 className={`text-2xl font-bold ${
                botData.metrics?.pnl >= 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                ${botData.metrics?.pnl?.toFixed(2) || '0.00'}
              </h3>
              <p className={`text-sm ${
                botData.metrics?.pnl_percentage >= 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                {botData.metrics?.pnl_percentage >= 0 ? '+' : ''}
                {botData.metrics?.pnl_percentage?.toFixed(2) || '0.00'}%
              </p>
            </div>
            <Activity className="h-8 w-8 text-[#87CEEB]" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500">Active Positions</p>
              <h3 className="text-2xl font-bold text-[#001F3F]">
                {botData.positions?.length || 0}
              </h3>
            </div>
            <LineChart className="h-8 w-8 text-[#87CEEB]" />
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Current Positions</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Quantity</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Entry Price</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Current Price</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">P/L %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {!botData.positions?.length ? (
                <tr>
                  <td colSpan="5" className="px-6 py-4 text-center text-gray-500">
                    No active positions
                  </td>
                </tr>
              ) : (
                botData.positions.map((position, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4">{position.symbol}</td>
                    <td className="px-6 py-4 text-right">
                      {parseFloat(position.quantity).toFixed(4)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      ${parseFloat(position.entry_price).toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      ${parseFloat(position.current_price).toFixed(2)}
                    </td>
                    <td className={`px-6 py-4 text-right ${
                      position.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {position.pnl >= 0 ? '+' : ''}
                      {parseFloat(position.pnl).toFixed(2)}%
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trade History */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Trades</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Price</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Quantity</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Value</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {!botData.trades?.length ? (
                <tr>
                  <td colSpan="6" className="px-6 py-4 text-center text-gray-500">
                    No trades yet
                  </td>
                </tr>
              ) : (
                botData.trades.map((trade, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4">
                      {new Date(trade.timestamp).toLocaleString()}
                    </td>
                    <td className="px-6 py-4">{trade.symbol}</td>
                    <td className={`px-6 py-4 ${
                      trade.type === 'buy' ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {trade.type.toUpperCase()}
                    </td>
                    <td className="px-6 py-4 text-right">
                      ${parseFloat(trade.price).toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {parseFloat(trade.quantity).toFixed(4)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      ${parseFloat(trade.value).toFixed(2)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BotDashboard;
