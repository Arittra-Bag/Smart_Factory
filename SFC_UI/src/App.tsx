import React, { useState } from 'react';
import Header from './components/shared/Header';
import HomePage from './components/HomePage';
import AdminDashboard from './components/admin/AdminDashboard';
import ProductionPage from './components/production/ProductionPage';

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'admin' | 'control'>('home');
  const [systemStatus] = useState<'Online' | 'Offline'>('Online');

  const handleNavigateToAdmin = () => setCurrentPage('admin');
  const handleNavigateToControl = () => setCurrentPage('control');

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
          />
        ) : currentPage === 'admin' ? (
          <AdminDashboard />
        ) : (
          <ProductionPage />
        )}
      </main>
    </div>
  );
}

export default App;
