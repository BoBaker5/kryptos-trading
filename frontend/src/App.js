import React, { useState } from 'react';
import DashboardLayout from './components/DashboardLayout';
import TradingDashboard from './components/TradingDashboard';
import DemoDashboard from './components/DemoDashboard';

function App() {
  const [currentView, setCurrentView] = useState('live');

  const navItems = [
    { id: 'live', label: 'Live Trading', icon: LayoutDashboard },
    { id: 'demo', label: 'Demo Account', icon: LineChart },
    { id: 'analytics', label: 'Analytics', icon: Activity },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  return (
    <DashboardLayout 
      currentView={currentView} 
      setCurrentView={setCurrentView}
      navItems={navItems}
    >
      {currentView === 'live' && <TradingDashboard />}
      {currentView === 'demo' && <DemoDashboard />}
      {currentView === 'analytics' && <div>Analytics Coming Soon</div>}
      {currentView === 'settings' && <div>Settings Coming Soon</div>}
    </DashboardLayout>
  );
}

export default App;
