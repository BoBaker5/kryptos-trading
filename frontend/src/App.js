// App.js
import React from 'react';
import DashboardLayout from './components/DashboardLayout';
import TradingDashboard from './components/TradingDashboard';

function App() {
  return (
    <DashboardLayout>
      <TradingDashboard />
    </DashboardLayout>
  );
}

export default App;
