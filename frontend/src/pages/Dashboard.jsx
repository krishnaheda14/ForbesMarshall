// src/pages/Dashboard.jsx
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Box,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  CloudUpload as LoadIcon,
  Insights as AIIcon,
  CheckCircle as CheckIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
import { loadData, getCurrentSchedule, getAIInsights, getDataInfo } from '../services/api';
import KPICards from '../components/KPICards';
import AIInsightsPanel from '../components/AIInsightsPanel';

function Dashboard() {
  const { enqueueSnackbar } = useSnackbar();
  const {
    dataLoaded,
    setDataLoaded,
    currentHeuristic,
    currentSchedule,
    setCurrentSchedule,
    loading,
    setLoading,
    metrics,
  } = useSchedulerStore();

  const [aiInsights, setAiInsights] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);

  // Check if data is already loaded on mount
  useEffect(() => {
    checkDataStatus();
  }, []);

  // Auto-fetch schedule when heuristic changes
  useEffect(() => {
    if (currentHeuristic && !currentSchedule) {
      fetchCurrentSchedule();
    }
  }, [currentHeuristic]);

  const checkDataStatus = async () => {
    try {
      const result = await getDataInfo();
      if (result.operations_count > 0) {
        setDataLoaded(true, {
          operations: result.operations_count,
          machines: result.machines_count,
          jobs: result.jobs_count
        });
      }
    } catch (error) {
      // Data not loaded yet, that's fine
    }
  };

  const fetchCurrentSchedule = async () => {
    try {
      const result = await getCurrentSchedule();
      if (result.schedule && result.schedule.length > 0) {
        setCurrentSchedule(result.schedule);
      }
    } catch (error) {
      // Expected if no schedule computed yet
    }
  };

  const handleGetAIInsights = async () => {
    if (!currentSchedule || !currentHeuristic) {
      enqueueSnackbar('Please compute and apply a heuristic first', {
        variant: 'warning',
      });
      return;
    }

    try {
      setLoadingAI(true);
      const currentMetrics = metrics[currentHeuristic] || {};
      const prompt = `Analyze the ${currentHeuristic} scheduling heuristic performance and provide insights:\n- Makespan: ${currentMetrics.Makespan_Days || 'N/A'} days\n- Tardiness: ${currentMetrics.Total_Tardiness_Days || 'N/A'} days\n- Utilization: ${currentMetrics['Machine_Utilization_%'] || 'N/A'}%\n- On-Time: ${currentMetrics['On_Time_%'] || 'N/A'}%\n\nProvide: 1) Performance analysis, 2) Bottlenecks, 3) Recommendations`;
      
      const result = await getAIInsights(prompt, {
        heuristic: currentHeuristic,
        metrics: currentMetrics
      });
      
      setAiInsights(result.insights);
      enqueueSnackbar('AI insights generated!', { variant: 'success' });
    } catch (error) {
      console.error('Failed to fetch AI insights:', error);
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoadingAI(false);
    }
  };

  const handleLoadData = async () => {
    try {
      setLoading(true);
      enqueueSnackbar('Loading dataset...', { variant: 'info' });
      
      const result = await loadData(50);
      
      setDataLoaded(true, result.stats);
      enqueueSnackbar('Data loaded successfully!', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };



  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h1" gutterBottom>
          CNC Scheduling Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Advanced production scheduling with AI-powered insights
        </Typography>
      </Box>

      {!dataLoaded ? (
        <Card sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Welcome to CNC Scheduler
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            Load your dataset to begin scheduling operations
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <LoadIcon />}
            onClick={handleLoadData}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Load Dataset'}
          </Button>
        </Card>
      ) : (
        <>
          {/* Dataset Loaded Status Card */}
          <Card sx={{ mb: 3, bgcolor: 'success.light' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box display="flex" alignItems="center" gap={2}>
                  <CheckIcon sx={{ fontSize: 40, color: 'success.dark' }} />
                  <Box>
                    <Typography variant="h6" sx={{ color: 'success.dark' }}>
                      Dataset Loaded Successfully
                    </Typography>
                    <Box display="flex" gap={1} mt={1}>
                      <Chip 
                        label={`${useSchedulerStore.getState().dataStats?.operations || 0} Operations`} 
                        size="small" 
                        color="success"
                      />
                      <Chip 
                        label={`${useSchedulerStore.getState().dataStats?.machines || 0} Machines`} 
                        size="small" 
                        color="success"
                      />
                      <Chip 
                        label={`${useSchedulerStore.getState().dataStats?.jobs || 0} Jobs`} 
                        size="small" 
                        color="success"
                      />
                    </Box>
                  </Box>
                </Box>
                <Button
                  variant="outlined"
                  color="success"
                  startIcon={<RefreshIcon />}
                  onClick={handleLoadData}
                  disabled={loading}
                >
                  Reload Dataset
                </Button>
              </Box>
            </CardContent>
          </Card>

          {!currentHeuristic ? (
            <Alert severity="info" sx={{ mb: 3 }}>
              Select a heuristic from the sidebar and click "Compute All Heuristics" to get started.
            </Alert>
          ) : (
            <>
              <Box sx={{ mb: 3 }}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="h6">
                        Active Heuristic: <strong>{currentHeuristic}</strong>
                      </Typography>
                      <Button
                        variant="outlined"
                        onClick={handleGetAIInsights}
                        disabled={loadingAI || !currentSchedule}
                        startIcon={loadingAI ? <CircularProgress size={20} /> : <AIIcon />}
                      >
                        {loadingAI ? 'Analyzing...' : 'Get AI Insights'}
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Box>

              <KPICards />

              {aiInsights && (
                <Box sx={{ mt: 3 }}>
                  <AIInsightsPanel insights={aiInsights} onClose={() => setAiInsights(null)} />
                </Box>
              )}

              <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Quick Actions
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={4}>
                          <Button
                            fullWidth
                            variant="outlined"
                            onClick={() => window.location.href = '/comparison'}
                          >
                            View Comparison
                          </Button>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <Button
                            fullWidth
                            variant="outlined"
                            onClick={() => window.location.href = '/gantt'}
                          >
                            View Gantt Chart
                          </Button>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <Button
                            fullWidth
                            variant="outlined"
                            onClick={() => window.location.href = '/operations'}
                          >
                            View Operations
                          </Button>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </>
          )}
        </>
      )}
    </Container>
  );
}

export default Dashboard;
