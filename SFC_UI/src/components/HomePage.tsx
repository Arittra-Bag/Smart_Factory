import React from 'react';
import { Factory, Shield, Zap, TrendingUp, Users, Clock, CheckCircle, ArrowRight, Play, BarChart3, Eye, Cpu } from 'lucide-react';

interface HomePageProps {
  onNavigateToAdmin: () => void;
  onNavigateToControl: () => void;
}

export default function HomePage({ onNavigateToAdmin, onNavigateToControl }: HomePageProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-r from-slate-800 via-blue-900 to-indigo-900 text-white">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-16">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="flex items-center space-x-3">
                <Factory className="h-8 w-8 text-blue-400" />
                <span className="text-blue-400 font-semibold">Smart Factory Control</span>
              </div>
              <h1 className="text-4xl lg:text-6xl font-bold leading-tight">
                Next-Generation
                <span className="block text-blue-400">Manufacturing</span>
                <span className="block">Intelligence</span>
              </h1>
              <p className="text-xl text-slate-300 leading-relaxed">
                Revolutionize your production line with AI-powered quality control, 
                real-time monitoring, and predictive analytics for the modern factory.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={onNavigateToControl}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center space-x-2 group"
                >
                  <Play className="h-5 w-5" />
                  <span>Start Production</span>
                  <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </button>
                <button
                  onClick={onNavigateToAdmin}
                  className="border-2 border-white/30 hover:border-white/50 text-white px-8 py-4 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center space-x-2"
                >
                  <BarChart3 className="h-5 w-5" />
                  <span>View Analytics</span>
                </button>
              </div>
            </div>
            <div className="relative">
              <div className="bg-gradient-to-br from-blue-600/20 to-indigo-600/20 rounded-2xl p-8 backdrop-blur-sm border border-white/10">
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-white/10 rounded-lg p-6 backdrop-blur-sm">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                      <span className="text-sm font-medium">Production Active</span>
                    </div>
                    <div className="text-2xl font-bold">1,247</div>
                    <div className="text-sm text-slate-300">Items Produced</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-6 backdrop-blur-sm">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                      <span className="text-sm font-medium">Quality Score</span>
                    </div>
                    <div className="text-2xl font-bold">98.5%</div>
                    <div className="text-sm text-slate-300">Defect Rate: 1.5%</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-6 backdrop-blur-sm">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                      <span className="text-sm font-medium">Uptime</span>
                    </div>
                    <div className="text-2xl font-bold">99.2%</div>
                    <div className="text-sm text-slate-300">System Availability</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-6 backdrop-blur-sm">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                      <span className="text-sm font-medium">Efficiency</span>
                    </div>
                    <div className="text-2xl font-bold">94.8%</div>
                    <div className="text-sm text-slate-300">Production Efficiency</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-slate-900 mb-4">
              Why Choose Smart Factory Control?
            </h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              Our comprehensive solution combines cutting-edge AI, real-time monitoring, 
              and predictive analytics to transform your manufacturing operations.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-8 border border-blue-100 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mb-6">
                <Eye className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Real-Time Monitoring</h3>
              <p className="text-slate-600 leading-relaxed">
                Monitor your production line in real-time with live video feeds, 
                quality metrics, and instant alerts for any anomalies.
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-8 border border-green-100 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center mb-6">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">AI Quality Control</h3>
              <p className="text-slate-600 leading-relaxed">
                Advanced computer vision and machine learning algorithms detect 
                defects with 99% accuracy, ensuring consistent product quality.
              </p>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-violet-50 rounded-xl p-8 border border-purple-100 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center mb-6">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Predictive Analytics</h3>
              <p className="text-slate-600 leading-relaxed">
                Leverage historical data and AI insights to predict maintenance needs, 
                optimize production schedules, and prevent costly downtime.
              </p>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl p-8 border border-orange-100 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-orange-600 rounded-lg flex items-center justify-center mb-6">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Gesture Control</h3>
              <p className="text-slate-600 leading-relaxed">
                Intuitive hand gesture recognition allows operators to control 
                production processes naturally without touching contaminated surfaces.
              </p>
            </div>

            <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-xl p-8 border border-red-100 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-red-600 rounded-lg flex items-center justify-center mb-6">
                <Clock className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">24/7 Operation</h3>
              <p className="text-slate-600 leading-relaxed">
                Continuous monitoring and automated quality checks ensure 
                consistent production quality around the clock.
              </p>
            </div>

            <div className="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl p-8 border border-teal-100 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-teal-600 rounded-lg flex items-center justify-center mb-6">
                <Cpu className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Smart Integration</h3>
              <p className="text-slate-600 leading-relaxed">
                Seamlessly integrate with existing manufacturing systems and 
                databases for comprehensive factory management.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-slate-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              Our intelligent system works in three simple steps to revolutionize your manufacturing process.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-2xl font-bold text-white">1</span>
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Setup & Integration</h3>
              <p className="text-slate-600 leading-relaxed">
                Connect our AI-powered cameras and sensors to your production line. 
                Our system automatically calibrates and begins monitoring your processes.
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-2xl font-bold text-white">2</span>
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Real-Time Analysis</h3>
              <p className="text-slate-600 leading-relaxed">
                Advanced AI algorithms analyze every product in real-time, detecting 
                defects, monitoring quality metrics, and providing instant feedback.
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-2xl font-bold text-white">3</span>
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-4">Optimize & Improve</h3>
              <p className="text-slate-600 leading-relaxed">
                Access comprehensive analytics and insights to optimize production, 
                reduce waste, and continuously improve your manufacturing processes.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-r from-slate-800 to-blue-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-blue-400 mb-2">99.5%</div>
              <div className="text-slate-300">Defect Detection Accuracy</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-green-400 mb-2">50%</div>
              <div className="text-slate-300">Reduction in Quality Issues</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-purple-400 mb-2">24/7</div>
              <div className="text-slate-300">Continuous Monitoring</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-orange-400 mb-2">30%</div>
              <div className="text-slate-300">Increase in Efficiency</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl lg:text-4xl font-bold text-slate-900 mb-6">
            Ready to Transform Your Manufacturing?
          </h2>
          <p className="text-xl text-slate-600 mb-8">
            Join the future of smart manufacturing with our AI-powered quality control system.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={onNavigateToControl}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center space-x-2"
            >
              <Play className="h-5 w-5" />
              <span>Start Free Demo</span>
            </button>
            <button
              onClick={onNavigateToAdmin}
              className="border-2 border-slate-300 hover:border-slate-400 text-slate-700 px-8 py-4 rounded-lg font-semibold transition-all duration-300"
            >
              View Live Demo
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <Factory className="h-6 w-6 text-blue-400" />
                <span className="text-lg font-bold">Smart Factory Control</span>
              </div>
              <p className="text-slate-400">
                Revolutionizing manufacturing with AI-powered quality control and real-time monitoring.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Features</h3>
              <ul className="space-y-2 text-slate-400">
                <li>Real-time Monitoring</li>
                <li>AI Quality Control</li>
                <li>Predictive Analytics</li>
                <li>Gesture Control</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Solutions</h3>
              <ul className="space-y-2 text-slate-400">
                <li>Manufacturing</li>
                <li>Quality Assurance</li>
                <li>Process Optimization</li>
                <li>Data Analytics</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Contact</h3>
              <ul className="space-y-2 text-slate-400">
                <li>support@smartfactory.com</li>
                <li>+1 (555) 123-4567</li>
                <li>24/7 Support Available</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-slate-800 mt-8 pt-8 text-center text-slate-400">
            <p>&copy; 2024 Smart Factory Control. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
} 