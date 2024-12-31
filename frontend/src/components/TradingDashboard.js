import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, DollarSign, TrendingUp } from 'lucide-react';

const API_URL = 'http://129.158.53.116:8000';
const USER_ID = 1; // For demo purposes

const TradingDashboard = () => {
  const [botData, setBotData] = useState({
    status: 'stopped',
    positions: [],
    portfolio_value: 0,
    daily_pnl: 0,
    performance: []
  });

  const [credentials, setCredentials] = useState({
    api_key: '',
    secret_key: ''
  });

  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchBotStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/bot-status/${USER_ID}`);
        setBotData(response.data);
      } catch (err) {
        console.error('Error fetching bot status:', err);
        setError(err.message);
      }
    };

    fetchBotStatus();
    const interval = setInterval(fetchBotStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleStartBot = async (e) => {
    e.preventDefault();
    try {
      setError(null);
      console.log('Starting bot with credentials:', credentials);
      await axios.post(`${API_URL}/start-bot/${USER_ID}`, credentials);
      setBotData(prev => ({ ...prev, status: 'running' }));
    } catch (err) {
      console.error('Error starting bot:', err);
      setError(err.response?.data?.detail || err.message);
    }
  };

  const handleStopBot = async () => {
    try {
      setError(null);
      await axios.post(`${API_URL}/stop-bot/${USER_ID}`);
      setBotData(prev => ({ ...prev, status: 'stopped' }));
    } catch (err) {
      console.error('Error stopping bot:', err);
      setError(err.response?.data?.detail || err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Crypto Trading Dashboard</h1>
          <div className="flex items-center mt-4">
            <span className={`px-3 py-1 rounded-full ${
              botData.status === 'running' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {botData.status === 'running' ? 'Bot Active' : 'Bot Stopped'}
            </span>
          </div>
        </header>

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4">
            {error}
          </div>
        )}

        {botData.status === 'stopped' && (
          <div className="bg-white p-6 rounded-lg shadow mb-8">
            <h2 className="text-xl font-bold mb-4">Start Trading Bot</h2>
            <form onSubmit={handleStartBot} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">API Key</label>
                <input
                  type="text"
                  value={credentials.api_key}
                  onChange={(e) => setCredentials(prev => ({...prev, api_key: e.target.value}))}
                  className="mt-1 block w-full rounded-md border border-gray-300 p-2"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Secret Key</label>
                <input
                  type="password"
                  value={credentials.secret_key}
                  onChange={(e) => setCredentials(prev => ({...prev, secret_key: e.target.value}))}
                  className="mt-1 block w-full rounded-md border border-gray-300 p-2"
                  required
                />
              </div>
              <button
                type="submit"
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg"
              >
                Start Bot
              </button>
            </form>
          </div>
        )}

        {botData.status === 'running' && (
          <button
            onClick={handleStopBot}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg mb-8"
          >
            Stop Bot
          </button>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <DollarSign className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-gray-500">Portfolio Value</p>
                <p className="text-2xl font-bold">${botData.portfolio_value.toFixed(2)}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-green-500" />
              <div className="ml-4">
                <p className="text-gray-500">24h Profit/Loss</p>
                <p className={`text-2xl font-bold ${botData.daily_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {botData.daily_pnl >= 0 ? '+' : ''}{botData.daily_pnl.toFixed(2)}%
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-purple-500" />
              <div className="ml-4">
                <p className="text-gray-500">Active Positions</p>
                <p className="text-2xl font-bold">{botData.positions.length}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow mb-8">
          <h2 className="text-xl font-bold mb-4">Active Positions</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Quantity</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Entry Price</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Current Price</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">P/L %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {botData.positions.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-4 text-center text-gray-500">
                      No active positions
                    </td>
                  </tr>
                ) : (
                  botData.positions.map((position, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4">{position.symbol}</td>
                      <td className="px-6 py-4 text-right">{position.quantity.toFixed(4)}</td>
                      <td className="px-6 py-4 text-right">${position.entry_price.toFixed(2)}</td>
                      <td className="px-6 py-4 text-right">${position.current_price.toFixed(2)}</td>
                      <td className={`px-6 py-4 text-right ${position.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
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
