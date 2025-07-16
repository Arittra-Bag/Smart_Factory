import React, { useState } from 'react';
import { Factory, User, Power, Menu, X, Search } from 'lucide-react';

interface HeaderProps {
  currentPage: 'home' | 'admin' | 'control' | 'detection';
  onPageChange: (page: 'home' | 'admin' | 'control' | 'detection') => void;
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
              onClick={() => onPageChange('home')}
              className="px-3 lg:px-4 py-2 rounded-md text-sm font-medium transition-colors text-slate-300 hover:text-white hover:bg-slate-700"
            >
              <span className="hidden lg:inline">Home</span>
              <span className="lg:hidden">Home</span>
            </button>
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
            <button
              onClick={() => onPageChange('detection')}
              className={`px-3 lg:px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                currentPage === 'detection'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              <span className="hidden lg:inline">Detection</span>
              <span className="lg:hidden">Detection</span>
            </button>
          </nav>

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
                  onPageChange('home');
                  setIsMobileMenuOpen(false);
                }}
                className="block w-full text-left px-4 py-2 rounded-md text-sm font-medium transition-colors text-slate-300 hover:text-white hover:bg-slate-700"
              >
                Home
              </button>
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
              <button
                onClick={() => {
                  onPageChange('detection');
                  setIsMobileMenuOpen(false);
                }}
                className={`block w-full text-left px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  currentPage === 'detection'
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                Detection
              </button>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}