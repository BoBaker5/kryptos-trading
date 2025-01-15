// src/components/DemoDashboard.js
import React, { useEffect, useState } from 'react';
import { LineChart, Activity, DollarSign, TrendingUp } from 'lucide-react';

const DemoDashboard = () => {
    const [status, setStatus] = useState(null);
    const [trades, setTrades] = useState([]);
    const [wsConnected, setWsConnected] = useState(false);
    const API_URL = process.env.REACT_APP_API_URL;

    useEffect(() => {
        // Fetch initial data
        const fetchData = async () => {
            try {
                const [statusRes, tradesRes] = await Promise.all([
                    fetch(`${API_URL}/bot/demo/status`).then(res => res.json()),
                    fetch(`${API_URL}/bot/demo/trades`).then(res => res.json())
                ]);
                
                if (statusRes.status === 'success') {
                    setStatus(statusRes.data);
                }
                if (tradesRes.status === 'success') {
                    setTrades(tradesRes.data);
                }
            } catch (error) {
                console.error('Error fetching bot data:', error);
            }
        };
        
        fetchData();

        // Set up WebSocket connection
        const wsUrl = API_URL.replace('http', 'ws');
        const ws = new WebSocket(`${wsUrl}/bot/demo`);
        
        ws.onopen = () => {
            setWsConnected(true);
            console.log('WebSocket Connected');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'status_update') {
                setStatus(data.data);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setWsConnected(false);
        };

        ws.onclose = () => {
            setWsConnected(false);
        };

        // Cleanup
        return () => {
            ws.close();
        };
    }, [API_URL]);

    if (!status) return (
        <div className="flex items-center justify-center h-screen">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#87CEEB]"></div>
        </div>
    );

    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-6 text-[#001F3F]">Demo Trading Dashboard</h1>
            
            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-500">Current Equity</p>
                            <h3 className="text-2xl font-bold">${status.metrics.current_equity.toFixed(2)}</h3>
                        </div>
                        <DollarSign className="h-8 w-8 text-[#87CEEB]" />
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-500">P&L</p>
                            <h3 className={`text-2xl font-bold ${status.metrics.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                ${status.metrics.pnl.toFixed(2)}
                            </h3>
                            <p className={`text-sm ${status.metrics.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                {status.metrics.pnl_percentage.toFixed(2)}%
                            </p>
                        </div>
                        <Activity className="h-8 w-8 text-[#87CEEB]" />
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-500">Status</p>
                            <div className="flex items-center mt-1">
                                <div className={`h-3 w-3 rounded-full ${status.isRunning ? 'bg-green-500' : 'bg-red-500'} mr-2`}></div>
                                <h3 className="text-lg font-medium">{status.isRunning ? 'Running' : 'Stopped'}</h3>
                            </div>
                        </div>
                        <LineChart className="h-8 w-8 text-[#87CEEB]" />
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-500">Balance</p>
                            <h3 className="text-2xl font-bold">${status.balance.ZUSD?.toFixed(2) || 0}</h3>
                        </div>
                        <TrendingUp className="h-8 w-8 text-[#87CEEB]" />
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
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Entry Price</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Current Price</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P&L</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {status.positions.length > 0 ? (
                                status.positions.map((position) => (
                                    <tr key={position.pair}>
                                        <td className="px-6 py-4 whitespace-nowrap">{position.pair}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">{position.vol}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">${parseFloat(position.entry_price).toFixed(4)}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">${parseFloat(position.price).toFixed(4)}</td>
                                        <td className={`px-6 py-4 whitespace-nowrap ${parseFloat(position.pnl) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                            ${parseFloat(position.pnl).toFixed(2)} ({parseFloat(position.pnl_percentage).toFixed(2)}%)
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="5" className="px-6 py-4 text-center text-gray-500">No active positions</td>
                                </tr>
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
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {trades.length > 0 ? (
                                trades.map((trade) => (
                                    <tr key={trade.timestamp}>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            {new Date(trade.timestamp).toLocaleString()}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">{trade.symbol}</td>
                                        <td className={`px-6 py-4 whitespace-nowrap ${trade.type === 'buy' ? 'text-green-500' : 'text-red-500'}`}>
                                            {trade.type.toUpperCase()}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">${trade.price}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">{trade.quantity}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">${trade.value}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="6" className="px-6 py-4 text-center text-gray-500">No trades yet</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default DemoDashboard;
