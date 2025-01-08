import React, { useState, useEffect } from 'react';
import { Activity, DollarSign, TrendingUp } from 'lucide-react';

// Use environment variables for API configuration
const API_URL = process.env.REACT_APP_API_URL || 'https://150.136.163.34';
const USER_ID = process.env.REACT_APP_USER_ID || 1;

// Improved API client setup
const createApiClient = () => {
  const handleRequest = async (endpoint, options = {}) => {
    const defaultOptions = {
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      timeout: 5000,
    };

    try {
      const response = await fetch(`${API_URL}${endpoint}`, {
        ...defaultOptions,
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw new Error('Unable to connect to trading server. Please check if the server is running and accessible.');
      }
      throw error;
    }
  };

  return {
    getBotStatus: () => handleRequest(`/bot-status/${USER_ID}`),
    startBot: (credentials) => handleRequest(`/start-bot/${USER_ID}`, {
      method: 'POST',
      body: JSON.stringify(credentials),
    }),
    stopBot: () => handleRequest(`/stop-bot/${USER_ID}`, {
      method: 'POST',
    }),
  };
};

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
  const [isLoading, setIsLoading] = useState(false);
  const api = createApiClient();

  useEffect(() => {
    let isMounted = true;
    const fetchBotStatus = async () => {
      try {
        setIsLoading(true);
        const data = await api.getBotStatus();
        if (isMounted) {
          setBotData(data);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          console.error('Error fetching bot status:', err);
          setError(err.message);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchBotStatus();
    const interval = setInterval(fetchBotStatus, 30000);
    
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleStartBot = async (e) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      setError(null);
      await api.startBot(credentials);
      setBotData(prev => ({ ...prev, status: 'running' }));
    } catch (err) {
      console.error('Error starting bot:', err);
      setError('Failed to start bot. Please check your credentials and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopBot = async () => {
    try {
      setIsLoading(true);
      setError(null);
      await api.stopBot();
      setBotData(prev => ({ ...prev, status: 'stopped' }));
    } catch (err) {
      console.error('Error stopping bot:', err);
      setError('Failed to stop bot. Please try again.');
    } finally {
      setIsLoading(false);
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

        {isLoading && (
          <div className="fixed top-4 right-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        )}

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
                disabled={isLoading}
              >
                {isLoading ? 'Starting...' : 'Start Bot'}
              </button>
            </form>
          </div>
        )}

        {botData.status === 'running' && (
          <button
            onClick={handleStopBot}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg mb-8"
            disabled={isLoading}
          >
            {isLoading ? 'Stopping...' : 'Stop Bot'}
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
