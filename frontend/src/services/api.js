// src/services/api.js
import axios from 'axios';
import { logAPICall } from '../components/APIDebugConsole';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    config.metadata = { startTime: new Date().getTime() };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    const duration = new Date().getTime() - response.config.metadata.startTime;
    
    // Handle different request data types
    let requestData = null;
    if (response.config.data) {
      if (response.config.data instanceof FormData) {
        requestData = { type: 'FormData', note: 'File upload or form data' };
      } else {
        try {
          requestData = JSON.parse(response.config.data);
        } catch {
          requestData = response.config.data;
        }
      }
    }
    
    logAPICall({
      method: response.config.method.toUpperCase(),
      url: response.config.url,
      status: response.status,
      duration: duration,
      timestamp: new Date().getTime(),
      request: requestData,
      response: response.data,
    });
    
    return response;
  },
  (error) => {
    const duration = error.config?.metadata?.startTime 
      ? new Date().getTime() - error.config.metadata.startTime 
      : 0;
    
    // Handle different request data types for errors
    let requestData = null;
    if (error.config?.data) {
      if (error.config.data instanceof FormData) {
        requestData = { type: 'FormData', note: 'File upload or form data' };
      } else {
        try {
          requestData = JSON.parse(error.config.data);
        } catch {
          requestData = error.config.data;
        }
      }
    }
    
    logAPICall({
      method: error.config?.method?.toUpperCase() || 'UNKNOWN',
      url: error.config?.url || 'UNKNOWN',
      status: error.response?.status || 0,
      duration: duration,
      timestamp: new Date().getTime(),
      request: requestData,
      response: error.response?.data || null,
      error: error.response?.data?.detail || error.message,
    });
    
    return Promise.reject(error);
  }
);

// Data Loading
export const loadData = async (sampleSize = null) => {
  const response = await api.post('/api/data/load', { sample_size: sampleSize });
  return response.data;
};

export const unloadData = async () => {
  const response = await api.post('/api/data/unload');
  return response.data;
};

export const getDataInfo = async () => {
  const response = await api.get('/api/data/info');
  return response.data;
};

// Scheduling
export const computeHeuristic = async (heuristic) => {
  const response = await api.post('/api/schedule/compute', { heuristic });
  return response.data;
};

export const computeAllHeuristics = async () => {
  const response = await api.post('/api/schedule/compute-all');
  return response.data;
};

export const applyHeuristic = async (heuristic) => {
  const response = await api.post('/api/schedule/apply', { heuristic });
  return response.data;
};

export const getCurrentSchedule = async () => {
  const response = await api.get('/api/schedule/current');
  return response.data;
};

export async function computeCPSATOptimal(objective_mode = 'min_weighted', alpha = 0.1, time_limit_seconds = 30) {
  const payload = { objective_mode, alpha, time_limit_seconds };
  const response = await api.post('/api/schedule/cpsat', payload);
  return response.data;
};

// Machine Operations
export const getMachineData = async () => {
  const response = await api.get('/api/data/machines');
  return response.data;
};

export const simulateBreakdown = async (machineId, startTime, duration) => {
  const response = await api.post('/api/machine/breakdown', {
    machine_id: machineId,
    start_time: startTime,
    duration: duration,
  });
  return response.data;
};

// Job Operations
export const updateJobPriority = async (jobId, priority) => {
  const response = await api.post('/api/job/priority', {
    job_id: jobId,
    priority: priority,
  });
  return response.data;
};

export const addJob = async (jobData) => {
  const response = await api.post('/api/job/add', jobData);
  return response.data;
};

export const deleteJob = async (jobId) => {
  const response = await api.delete(`/api/job/${jobId}`);
  return response.data;
};

// Outsourcing
export const updateOutsourcingPolicy = async (costThreshold) => {
  const response = await api.post('/api/outsourcing/policy', {
    cost_threshold: costThreshold,
  });
  return response.data;
};

// AI Insights
export const getAIInsights = async (prompt, contextData = null) => {
  const response = await api.post('/api/ai/insights', {
    prompt,
    context_data: contextData,
  });
  return response.data;
};

// Metrics & Logs
export const getMetricsComparison = async () => {
  const response = await api.get('/api/metrics/comparison');
  return response.data;
};

export const getActivityLog = async () => {
  const response = await api.get('/api/activity-log');
  return response.data;
};

export default api;
