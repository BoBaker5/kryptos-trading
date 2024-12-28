// App.js
import React, { useState } from 'react';
import Header from './components/Header';
import TradingDashboard from './components/TradingDashboard';
import DemoDashboard from './components/DemoDashboard';
import { LayoutDashboard, LineChart, Settings, Activity } from 'lucide-react';

function App() {
  const [currentView, setCurrentView] = useState('live');

  const navItems = [
    { id: 'live', label: 'Live Trading', icon: LayoutDashboard },
    { id: 'demo', label: 'Demo Account', icon: LineChart },
    { id: 'analytics', label: 'Analytics', icon: Activity },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <div className="w-64 bg-[#001F3F] fixed h-full">
        <div className="flex items-center px-6 py-4">
          <img src="/logo.svg" alt="Kryptos" className="h-10" />
          <span className="ml-2 text-[#87CEEB] font-bold text-xl">KRYPTOS</span>
        </div>
        
        <nav className="mt-8">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setCurrentView(item.id)}
              className={`w-full flex items-center px-6 py-3 text-sm font-medium transition-colors
                ${currentView === item.id 
                  ? 'text-[#87CEEB] bg-[#87CEEB]/10' 
                  : 'text-gray-400 hover:text-[#87CEEB] hover:bg-[#87CEEB]/5'}`}
            >
              <item.icon className="h-5 w-5 mr-3" />
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div className="ml-64 flex-1 bg-gray-50">
        <Header 
          title={currentView === 'live' ? 'Live Trading Dashboard' : 'Demo Trading Dashboard'} 
        />
        
        <main className="p-8">
          {currentView === 'live' && <TradingDashboard />}
          {currentView === 'demo' && <DemoDashboard />}
          {currentView === 'analytics' && <div>Analytics Coming Soon</div>}
          {currentView === 'settings' && <div>Settings Coming Soon</div>}
        </main>
      </div>
    </div>
  );
}

export default App;