import React, { useState } from 'react';
import { Search, Filter, Download, Edit, Trash2, ChevronLeft, ChevronRight } from 'lucide-react';
import { SafetyRecord } from '../../types';

interface SafetyRecordsTableProps {
  records: SafetyRecord[];
}

export default function SafetyRecordsTable({ records }: SafetyRecordsTableProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDate, setFilterDate] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'Pass' | 'Fail'>('all');
  const [currentPage, setCurrentPage] = useState(1);
  const recordsPerPage = 5;

  const filteredRecords = records.filter(record => {
    const matchesSearch = record.date.includes(searchTerm) || 
                         record.batchSize.toString().includes(searchTerm) ||
                         record.defectRate.toString().includes(searchTerm);
    const matchesDate = !filterDate || record.date === filterDate;
    const matchesStatus = filterStatus === 'all' || record.status === filterStatus;
    
    return matchesSearch && matchesDate && matchesStatus;
  });

  const totalPages = Math.ceil(filteredRecords.length / recordsPerPage);
  const startIndex = (currentPage - 1) * recordsPerPage;
  const paginatedRecords = filteredRecords.slice(startIndex, startIndex + recordsPerPage);

  const uniqueDates = [...new Set(records.map(record => record.date))];

  const handleExportCSV = () => {
    const headers = ['DATE', 'BATCH SIZE', 'DEFECT RATE', 'TIMESTAMP', 'QUALITY SCORE', 'STATUS'];
    const rows = filteredRecords.map(r => [
      r.date, r.batchSize, r.defectRate, r.timestamp, r.qualityScore, r.status
    ]);
    const csvContent = [headers, ...rows]
      .map(e => e.join(','))
      .join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `safety_check_records.csv`;
    link.click();
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6 mb-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 space-y-3 sm:space-y-0">
        <h3 className="text-lg sm:text-xl font-semibold text-gray-900">Safety Check Records</h3>
        <button className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors w-full sm:w-auto text-sm sm:text-base" onClick={handleExportCSV}>
          <Download className="h-4 w-4" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <input
            type="text"
            placeholder="Search records..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          />
        </div>
        <select
          value={filterDate}
          onChange={(e) => setFilterDate(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
        >
          <option value="">All Dates</option>
          {uniqueDates.map(date => (
            <option key={date} value={date}>{date}</option>
          ))}
        </select>
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value as 'all' | 'Pass' | 'Fail')}
          className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
        >
          <option value="all">All Status</option>
          <option value="Pass">Pass</option>
          <option value="Fail">Fail</option>
        </select>
        <div className="flex items-center space-x-2 justify-center sm:justify-start">
          <Filter className="h-4 w-4 text-gray-400" />
          <span className="text-sm text-gray-600">{filteredRecords.length} records</span>
        </div>
      </div>

      {/* Table - Desktop */}
      <div className="hidden lg:block overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Batch Size</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Defect Rate</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality Score</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {paginatedRecords.map((record) => (
              <tr key={record.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{record.date}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{record.batchSize}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`text-sm font-medium ${
                    record.defectRate <= 2 ? 'text-green-600' : 
                    record.defectRate <= 5 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {record.defectRate}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{record.timestamp}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`text-sm font-medium ${
                    record.status === 'Pass' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {record.qualityScore}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                    record.status === 'Pass' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {record.status}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  <div className="flex space-x-2">
                    <button className="text-blue-600 hover:text-blue-800">
                      <Edit className="h-4 w-4" />
                    </button>
                    <button className="text-red-600 hover:text-red-800">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Cards - Mobile/Tablet */}
      <div className="lg:hidden space-y-4">
        {paginatedRecords.map((record) => (
          <div key={record.id} className="bg-gray-50 rounded-lg p-4 border hover:shadow-md transition-shadow duration-200">
            <div className="flex justify-between items-start mb-3">
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-gray-900 text-base">{record.date}</div>
                <div className="text-sm text-gray-500">{record.timestamp}</div>
              </div>
              <span className={`ml-2 px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full flex-shrink-0 ${
                record.status === 'Pass' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {record.status}
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-white rounded-lg p-3 border">
                <div className="text-xs text-gray-500 uppercase tracking-wide font-medium">Batch Size</div>
                <div className="font-bold text-lg text-gray-900">{record.batchSize}</div>
              </div>
              <div className="bg-white rounded-lg p-3 border">
                <div className="text-xs text-gray-500 uppercase tracking-wide font-medium">Defect Rate</div>
                <div className={`font-bold text-lg ${
                  record.defectRate <= 2 ? 'text-green-600' : 
                  record.defectRate <= 5 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {record.defectRate}%
                </div>
              </div>
              <div className="bg-white rounded-lg p-3 border">
                <div className="text-xs text-gray-500 uppercase tracking-wide font-medium">Quality Score</div>
                <div className={`font-bold text-lg ${
                  record.status === 'Pass' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {record.qualityScore}%
                </div>
              </div>
            </div>
            
            <div className="flex justify-end space-x-2 pt-2 border-t border-gray-200">
              <button className="text-blue-600 hover:text-blue-800 p-2 rounded-md hover:bg-blue-50 transition-colors">
                <Edit className="h-4 w-4" />
              </button>
              <button className="text-red-600 hover:text-red-800 p-2 rounded-md hover:bg-red-50 transition-colors">
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mt-6 space-y-3 sm:space-y-0">
        <div className="text-sm text-gray-700 text-center sm:text-left">
          Showing {startIndex + 1} to {Math.min(startIndex + recordsPerPage, filteredRecords.length)} of {filteredRecords.length} results
        </div>
        <div className="flex items-center justify-center space-x-2">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="p-2 border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="px-3 py-2 text-sm text-gray-700">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="p-2 border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}