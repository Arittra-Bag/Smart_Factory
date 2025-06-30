import React, { useState } from 'react';
import Header from './components/shared/Header';
import AdminDashboard from './components/admin/AdminDashboard';
import ProductionPage from './components/production/ProductionPage';

function App() {
  const [currentPage, setCurrentPage] = useState<'admin' | 'control'>('admin');
  const [systemStatus] = useState<'Online' | 'Offline'>('Online');

  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        systemStatus={systemStatus}
      />
      
      <main className="pb-4 sm:pb-6">
        {currentPage === 'admin' ? (
          <AdminDashboard />
        ) : (
          <ProductionPage />
        )}
      </main>
    </div>
  );
}

export default App;
