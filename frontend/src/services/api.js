// src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Data Loading
export const loadData = async (sampleSize = null) => {
  const response = await api.post('/api/data/load', { sample_size: sampleSize });
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
