// src/store/useSchedulerStore.js
import { create } from 'zustand';

const useSchedulerStore = create((set) => ({
  // Data state
  dataLoaded: false,
  dataStats: null,
  
  // Schedule state
  currentHeuristic: null,
  currentSchedule: null,
  schedules: {},
  metrics: {},
  
  // UI state
  loading: false,
  error: null,
  
  // Activity log
  activityLog: [],
  
  // Actions
  setDataLoaded: (loaded, stats = null) => set({ dataLoaded: loaded, dataStats: stats }),
  
  setCurrentHeuristic: (heuristic) => set({ currentHeuristic: heuristic, currentSchedule: null }),
  
  setCurrentSchedule: (schedule) => set({ currentSchedule: schedule }),
  
  setSchedules: (schedules) => set({ schedules }),
  
  setMetrics: (metrics) => set({ metrics }),
  
  setLoading: (loading) => set({ loading }),
  
  setError: (error) => set({ error }),
  
  setActivityLog: (log) => set({ activityLog: log }),
  
  addActivity: (activity) => set((state) => ({
    activityLog: [...state.activityLog, activity]
  })),
  
  clearError: () => set({ error: null }),
  
  reset: () => set({
    dataLoaded: false,
    dataStats: null,
    currentHeuristic: null,
    currentSchedule: null,
    schedules: {},
    metrics: {},
    loading: false,
    error: null,
    activityLog: []
  })
}));

export default useSchedulerStore;
