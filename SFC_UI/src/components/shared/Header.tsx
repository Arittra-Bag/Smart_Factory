import React, { useState } from 'react';
import { Factory, User, Power, Menu, X } from 'lucide-react';

interface HeaderProps {
  currentPage: 'admin' | 'control';
  onPageChange: (page: 'admin' | 'control') => void;
  systemStatus: 'Online' | 'Offline';
}

export default function Header({ currentPage, onPageChange, systemStatus }: HeaderProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <header className="bg-slate-800 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3 flex-shrink-0">
            <Factory className="h-6 w-6 sm:h-8 sm:w-8 text-blue-400" />
            <div className="hidden sm:block">
              <h1 className="text-lg sm:text-xl font-bold">Smart Factory Control</h1>
              <p className="text-xs text-slate-300 hidden lg:block">Industrial Manufacturing System</p>
            </div>
            <div className="sm:hidden">
              <h1 className="text-lg font-bold">Factory Control</h1>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-1">
            <button
              onClick={() => onPageChange('admin')}
              className={`px-3 lg:px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                currentPage === 'admin'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              <span className="hidden lg:inline">Admin Dashboard</span>
              <span className="lg:hidden">Admin</span>
            </button>
            <button
              onClick={() => onPageChange('control')}
              className={`px-3 lg:px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                currentPage === 'control'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              <span className="hidden lg:inline">Production Control</span>
              <span className="lg:hidden">Control</span>
            </button>
          </nav>

          {/* Status and User - Desktop */}
          <div className="hidden sm:flex items-center space-x-2 lg:space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                systemStatus === 'Online' ? 'bg-green-400' : 'bg-red-400'
              }`} />
              <span className="text-xs lg:text-sm">{systemStatus}</span>
            </div>
            <div className="flex items-center space-x-2">
              <User className="h-4 w-4 lg:h-5 lg:w-5" />
              <span className="text-xs lg:text-sm hidden lg:inline">Admin</span>
              <Power className="h-3 w-3 lg:h-4 lg:w-4 text-slate-400 hover:text-white cursor-pointer" />
            </div>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-md text-slate-300 hover:text-white hover:bg-slate-700"
          >
            {isMobileMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t border-slate-700 py-4">
            <div className="space-y-2">
              <button
                onClick={() => {
                  onPageChange('admin');
                  setIsMobileMenuOpen(false);
                }}
                className={`block w-full text-left px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  currentPage === 'admin'
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                Admin Dashboard
              </button>
              <button
                onClick={() => {
                  onPageChange('control');
                  setIsMobileMenuOpen(false);
                }}
                className={`block w-full text-left px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  currentPage === 'control'
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                Production Control
              </button>
            </div>
            
            {/* Mobile Status */}
            <div className="mt-4 pt-4 border-t border-slate-700">
              <div className="flex items-center justify-between px-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    systemStatus === 'Online' ? 'bg-green-400' : 'bg-red-400'
                  }`} />
                  <span className="text-sm">{systemStatus}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <User className="h-4 w-4" />
                  <span className="text-sm">Admin</span>
                  <Power className="h-4 w-4 text-slate-400 hover:text-white cursor-pointer" />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}