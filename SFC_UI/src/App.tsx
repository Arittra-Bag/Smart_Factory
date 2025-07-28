import React, { useState } from 'react';
import Header from './components/shared/Header';
import HomePage from './components/HomePage';
import AdminDashboard from './components/admin/AdminDashboard';
import ProductionPage from './components/production/ProductionPage';
import DetectionPage from './components/DetectionPage';
import AiCopilot from './components/shared/AiCopilot';

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'admin' | 'control' | 'detection' | 'copilot'>('home');
  const [systemStatus] = useState<'Online' | 'Offline'>('Online');

  const handleNavigateToAdmin = () => setCurrentPage('admin');
  const handleNavigateToControl = () => setCurrentPage('control');
  const handleNavigateToCopilot = () => setCurrentPage('copilot');

  return (
    <div className="min-h-screen bg-gray-50">
      {currentPage !== 'home' && (
        <Header 
          currentPage={currentPage}
          onPageChange={setCurrentPage}
          systemStatus={systemStatus}
        />
      )}
      
      <main className={currentPage !== 'home' ? 'pb-4 sm:pb-6' : ''}>
        {currentPage === 'home' ? (
          <HomePage 
            onNavigateToAdmin={handleNavigateToAdmin}
            onNavigateToControl={handleNavigateToControl}
            onNavigateToCopilot={handleNavigateToCopilot}
          />
        ) : currentPage === 'admin' ? (
          <AdminDashboard />
        ) : currentPage === 'control' ? (
          <ProductionPage />
        ) : currentPage === 'copilot' ? (
          <AiCopilot />
        ) : (
          <DetectionPage />
        )}
      </main>
    </div>
  );
}

export default App;
