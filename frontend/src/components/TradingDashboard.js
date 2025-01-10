// TradingDashboard.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, DollarSign, TrendingUp } from 'lucide-react';
import PerformanceChart from './PerformanceChart';
import ConfidenceIndicator from './ConfidenceIndicator';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const USER_ID = 1;

const TradingDashboard = () => {
  const [botData, setBotData] = useState({
    status: 'stopped',
    positions: [],
    portfolio_value: 0,
    daily_pnl: 0,
    returns_data: [],
    signals: []
  });

  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBotStatus = async () => {
      try {
        console.log('Fetching from:', `${API_URL}/bot-status/${USER_ID}`);
        const response = await axios.get(`${API_URL}/bot-status/${USER_ID}`);
        console.log('Response:', response.data);
        if (response.status === 200) {
          setBotData(response.data);
          setError(null);
        }
      } catch (err) {
        console.error('Error fetching bot status:', err);
        setError('Unable to connect to trading server. Please check your connection.');
      } finally {
        setLoading(false);
      }
    };

    fetchBotStatus();
    const interval = setInterval(fetchBotStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleStopBot = async () => {
    try {
      const response = await axios.post(`${API_URL}/stop-bot/${USER_ID}`);
      if (response.status === 200) {
        setBotData(prev => ({ ...prev, status: 'stopped' }));
      }
    } catch (err) {
      setError('Failed to stop bot. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className={styles.mainContainer}>
        <div className={styles.contentContainer}>
          <div className="flex justify-center items-center h-64">
            <span className="text-[#87CEEB]">Loading dashboard data...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.mainContainer}>
      <div className={styles.contentContainer}>
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-[#87CEEB]">Crypto Trading Dashboard</h1>
          <div className="flex items-center justify-between mt-4">
            <div className="flex items-center space-x-4">
              <span className={botData.status === 'running' ? styles.statusBadge.active : styles.statusBadge.inactive}>
                {botData.status === 'running' ? 'Bot Active' : 'Bot Stopped'}
              </span>
            </div>
            {botData.status === 'running' && (
              <button
                onClick={handleStopBot}
                className={styles.stopButton}
              >
                Stop Bot
              </button>
            )}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded">
            {error}
          </div>
        )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-[#001F3F] p-6 rounded-lg">
          <div className="flex items-center">
            <DollarSign className="h-8 w-8 text-[#87CEEB]" />
            <div className="ml-4">
              <p className="text-gray-400">Portfolio Value</p>
              <p className="text-2xl font-bold text-[#87CEEB]">
                ${botData.portfolio_value.toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-[#001F3F] p-6 rounded-lg">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-[#87CEEB]" />
            <div className="ml-4">
              <p className="text-gray-400">24h Profit/Loss</p>
              <p className={`text-2xl font-bold ${
                botData.daily_pnl >= 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                {botData.daily_pnl >= 0 ? '+' : ''}{botData.daily_pnl.toFixed(2)}%
              </p>
            </div>
          </div>
        </div>

        <div className="bg-[#001F3F] p-6 rounded-lg">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-[#87CEEB]" />
            <div className="ml-4">
              <p className="text-gray-400">Active Positions</p>
              <p className="text-2xl font-bold text-[#87CEEB]">
                {botData.positions.length}
              </p>
            </div>
          </div>
        </div>
      </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">
          <PerformanceChart data={botData.performance_data} />
          <ConfidenceIndicator signals={botData.signals} />
        </div>

        {/* Positions Table */}
        <div className="bg-[#001F3F] p-6 rounded-lg">
          <h2 className="text-xl font-bold text-[#87CEEB] p-6">Active Positions</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-[#003366]">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Symbol</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase">Quantity</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase">Entry Price</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase">Current Price</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase">P/L %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#003366]">
                {botData.positions.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-4 text-center text-gray-400">
                      No active positions
                    </td>
                  </tr>
                ) : (
                  botData.positions.map((position, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 text-[#87CEEB]">{position.symbol}</td>
                      <td className="px-6 py-4 text-right text-[#87CEEB]">
                        {position.quantity.toFixed(4)}
                      </td>
                      <td className="px-6 py-4 text-right text-[#87CEEB]">
                        ${position.entry_price.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 text-right text-[#87CEEB]">
                        ${position.current_price.toFixed(2)}
                      </td>
                      <td className={`px-6 py-4 text-right ${
                        position.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                      }`}>
                        {position.pnl >= 0 ? '+' : ''}{position.pnl.toFixed(2)}%
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;
