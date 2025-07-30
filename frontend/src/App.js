import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Sample bin data for demonstration
const sampleBins = [
  {
    bin_id: "BIN_001",
    fill_level_litres: 750,
    total_litres: 1000,
    fill_percentage: 75,
    location: "Downtown Plaza",
    latitude: 40.7128,
    longitude: -74.0060,
    temperature: 22,
    battery_level: 0.8
  },
  {
    bin_id: "BIN_002", 
    fill_level_litres: 900,
    total_litres: 1000,
    fill_percentage: 90,
    location: "Central Park",
    latitude: 40.7829,
    longitude: -73.9654,
    temperature: 24,
    battery_level: 0.9
  },
  {
    bin_id: "BIN_003",
    fill_level_litres: 300,
    total_litres: 1000,
    fill_percentage: 30,
    location: "University District",
    latitude: 40.7589,
    longitude: -73.9851,
    temperature: 20,
    battery_level: 0.7
  },
  {
    bin_id: "BIN_004",
    fill_level_litres: 850,
    total_litres: 1000,
    fill_percentage: 85,
    location: "Shopping Mall",
    latitude: 40.7505,
    longitude: -73.9934,
    temperature: 25,
    battery_level: 0.95
  },
  {
    bin_id: "BIN_005",
    fill_level_litres: 450,
    total_litres: 1000,
    fill_percentage: 45,
    location: "Residential Area",
    latitude: 40.7282,
    longitude: -73.7949,
    temperature: 21,
    battery_level: 0.6
  }
];

const Dashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [route, setRoute] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('predictions');

  useEffect(() => {
    fetchModelInfo();
    fetchAnalytics();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API}/model-info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await axios.get(`${API}/analytics`);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    }
  };

  const runBulkPredictions = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/predict-bulk`, sampleBins);
      setPredictions(response.data.predictions);
      setAnalytics(response.data.summary);
    } catch (error) {
      console.error('Error running predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const optimizeRoute = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/optimize-route`, {
        bins: sampleBins,
        start_location: { latitude: 40.7128, longitude: -74.0060 }
      });
      setRoute(response.data.optimized_route);
    } catch (error) {
      console.error('Error optimizing route:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (needsCollection, probability) => {
    if (needsCollection) return 'text-red-600 bg-red-50';
    if (probability > 0.7) return 'text-yellow-600 bg-yellow-50';
    return 'text-green-600 bg-green-50';
  };

  const getStatusText = (needsCollection, probability) => {
    if (needsCollection) return 'Needs Collection';
    if (probability > 0.7) return 'Monitor';
    return 'OK';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                🗑️ AI Trash Bin Management System
              </h1>
              <p className="text-gray-600">Smart waste collection optimization</p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={runBulkPredictions}
                disabled={loading}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {loading ? 'Processing...' : 'Run Predictions'}
              </button>
              <button
                onClick={optimizeRoute}
                disabled={loading}
                className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {loading ? 'Processing...' : 'Optimize Route'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-md flex items-center justify-center">
                  📊
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Bins</p>
                <p className="text-2xl font-semibold text-gray-900">{sampleBins.length}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-red-100 rounded-md flex items-center justify-center">
                  🚨
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Need Collection</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analytics ? analytics.bins_needing_collection || 0 : '-'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-100 rounded-md flex items-center justify-center">
                  📈
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Collection Rate</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analytics ? `${(analytics.collection_rate * 100).toFixed(1)}%` : '-'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-100 rounded-md flex items-center justify-center">
                  🤖
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Model Status</p>
                <p className="text-sm font-semibold text-gray-900">
                  {modelInfo ? (modelInfo.model_loaded ? '✅ Active' : '⚠️ Fallback') : '...'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6">
              {[
                { id: 'predictions', name: 'Bin Predictions', icon: '🎯' },
                { id: 'route', name: 'Collection Route', icon: '🗺️' },
                { id: 'model', name: 'Model Info', icon: '🤖' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.name}</span>
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {/* Predictions Tab */}
            {activeTab === 'predictions' && (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Bin Status Predictions</h3>
                  <span className="text-sm text-gray-500">
                    {predictions.length > 0 ? `${predictions.length} predictions` : 'Click "Run Predictions" to start'}
                  </span>
                </div>

                {predictions.length > 0 ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {predictions.map((prediction, index) => (
                      <div key={prediction.bin_id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex justify-between items-start mb-3">
                          <div>
                            <h4 className="font-semibold text-gray-900">{prediction.bin_id}</h4>
                            <p className="text-sm text-gray-600">
                              {sampleBins.find(b => b.bin_id === prediction.bin_id)?.location}
                            </p>
                          </div>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(prediction.needs_collection, prediction.probability)}`}>
                            {getStatusText(prediction.needs_collection, prediction.probability)}
                          </span>
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-500">Fill Level:</span>
                            <span className="font-medium">
                              {sampleBins.find(b => b.bin_id === prediction.bin_id)?.fill_percentage}%
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-500">Prediction Confidence:</span>
                            <span className="font-medium">{(prediction.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-500">Model Used:</span>
                            <span className="font-medium text-xs">{prediction.model_used}</span>
                          </div>
                        </div>

                        {/* Progress bar for fill level */}
                        <div className="mt-3">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${
                                prediction.needs_collection ? 'bg-red-500' : 
                                prediction.probability > 0.7 ? 'bg-yellow-500' : 'bg-green-500'
                              }`}
                              style={{ width: `${sampleBins.find(b => b.bin_id === prediction.bin_id)?.fill_percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-4xl mb-4">🎯</div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Predictions Yet</h3>
                    <p className="text-gray-600">Click "Run Predictions" to analyze all bins</p>
                  </div>
                )}
              </div>
            )}

            {/* Route Tab */}
            {activeTab === 'route' && (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Optimized Collection Route</h3>
                  <span className="text-sm text-gray-500">
                    {route.length > 0 ? `${route.length} stops optimized` : 'Click "Optimize Route" to plan'}
                  </span>
                </div>

                {route.length > 0 ? (
                  <div className="space-y-4">
                    <div className="bg-blue-50 rounded-lg p-4 mb-6">
                      <h4 className="font-medium text-blue-900 mb-2">Route Summary</h4>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-blue-700">Total Stops:</span>
                          <span className="font-semibold ml-2">{route.length}</span>
                        </div>
                        <div>
                          <span className="text-blue-700">Estimated Distance:</span>
                          <span className="font-semibold ml-2">
                            {route.length > 0 ? '12.5 km' : '0 km'}
                          </span>
                        </div>
                        <div>
                          <span className="text-blue-700">Estimated Time:</span>
                          <span className="font-semibold ml-2">
                            {route.length > 0 ? '45 min' : '0 min'}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      {route.map((stop, index) => (
                        <div key={stop.bin_id} className="flex items-center p-4 border rounded-lg hover:bg-gray-50">
                          <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                            {stop.route_order}
                          </div>
                          <div className="ml-4 flex-1">
                            <div className="flex justify-between items-start">
                              <div>
                                <h4 className="font-semibold text-gray-900">{stop.bin_id}</h4>
                                <p className="text-sm text-gray-600">{stop.location}</p>
                                <p className="text-xs text-gray-500">
                                  {stop.latitude?.toFixed(4)}, {stop.longitude?.toFixed(4)}
                                </p>
                              </div>
                              <div className="text-right">
                                <div className="text-sm font-medium text-gray-900">
                                  {stop.fill_percentage}% Full
                                </div>
                                <div className="text-xs text-gray-500">
                                  Priority Collection
                                </div>
                              </div>
                            </div>
                          </div>
                          {index < route.length - 1 && (
                            <div className="ml-4 text-gray-400">
                              ⬇️
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-4xl mb-4">🗺️</div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Route Planned</h3>
                    <p className="text-gray-600">Click "Optimize Route" to generate efficient collection path</p>
                  </div>
                )}
              </div>
            )}

            {/* Model Info Tab */}
            {activeTab === 'model' && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Machine Learning Model Information</h3>
                
                {modelInfo ? (
                  <div className="space-y-6">
                    {/* Model Status */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">Model Status</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <span className="text-sm text-gray-500">Status:</span>
                          <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                            modelInfo.model_loaded 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {modelInfo.model_loaded ? 'ML Model Active' : 'Fallback Mode'}
                          </span>
                        </div>
                        {modelInfo.model_loaded && (
                          <div>
                            <span className="text-sm text-gray-500">Algorithm:</span>
                            <span className="ml-2 font-medium">{modelInfo.model_name}</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Performance Metrics */}
                    {modelInfo.model_loaded && modelInfo.performance_metrics && (
                      <div>
                        <h4 className="font-medium text-gray-900 mb-3">Performance Metrics</h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                          {Object.entries(modelInfo.performance_metrics).map(([metric, value]) => (
                            <div key={metric} className="bg-white border rounded-lg p-4">
                              <div className="text-sm text-gray-500 capitalize">
                                {metric.replace('_', ' ').replace('-', ' ')}
                              </div>
                              <div className="text-xl font-semibold text-gray-900">
                                {(value * 100).toFixed(1)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Features */}
                    {modelInfo.model_loaded && modelInfo.features && (
                      <div>
                        <h4 className="font-medium text-gray-900 mb-3">Model Features</h4>
                        <div className="bg-white border rounded-lg p-4">
                          <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                            {modelInfo.features.map((feature, index) => (
                              <span key={index} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                {feature.replace('_', ' ').toLowerCase()}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Model Description */}
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">About the Model</h4>
                      <div className="bg-white border rounded-lg p-4">
                        <p className="text-gray-700 leading-relaxed">
                          {modelInfo.model_loaded 
                            ? `This system uses a ${modelInfo.model_name} algorithm trained on historical trash bin data to predict when bins need collection. The model analyzes factors like fill level, location, time patterns, and environmental conditions to optimize waste collection routes.`
                            : 'Currently using rule-based fallback predictions. The system predicts bins need collection when fill percentage exceeds 80%. For enhanced accuracy, please ensure the ML model file is available.'
                          }
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="text-gray-500 mt-2">Loading model information...</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;