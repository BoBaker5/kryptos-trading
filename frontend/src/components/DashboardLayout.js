// components/DashboardLayout.js
import React from 'react';
import { LayoutDashboard, LineChart, Activity, Settings } from 'lucide-react';

const DashboardLayout = ({ children }) => {
  const navItems = [
    { id: 'dashboard', label: 'Live Trading', icon: LayoutDashboard },
    { id: 'demo', label: 'Demo Account', icon: LineChart },
    { id: 'analytics', label: 'Analytics', icon: Activity },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <div className="w-64 bg-[#001F3F] fixed h-full">
        <div className="flex items-center p-6">
          <img src="/logo.svg" alt="Kryptos" className="h-12" />
          <span className="ml-2 text-[#87CEEB] font-bold text-xl">KRYPTOS</span>
        </div>
        
        <nav className="mt-8">
          {navItems.map(item => (
            <button
              key={item.id}
              className="w-full flex items-center px-6 py-3 text-gray-400 hover:text-[#87CEEB] hover:bg-[#87CEEB]/5"
            >
              <item.icon className="h-5 w-5 mr-3" />
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div className="ml-64 flex-1 bg-gray-50">
        <div className="p-8">
          {children}
        </div>
      </div>
    </div>
  );
};

export default DashboardLayout;
