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
} from '@mui/material';
import {
  CloudUpload as LoadIcon,
  Insights as AIIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
import { loadData, getCurrentSchedule, getAIInsights } from '../services/api';
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
  } = useSchedulerStore();

  const [aiInsights, setAiInsights] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);

  useEffect(() => {
    if (currentHeuristic && !currentSchedule) {
      fetchCurrentSchedule();
    }
  }, [currentHeuristic]);

  const fetchCurrentSchedule = async () => {
    try {
      const result = await getCurrentSchedule();
      setCurrentSchedule(result.schedule);
    } catch (error) {
      // Expected if no schedule computed yet
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

  const handleGetAIInsights = async () => {
    if (!currentSchedule || !currentHeuristic) {
      enqueueSnackbar('Please compute and apply a heuristic first', {
        variant: 'warning',
      });
      return;
    }

    try {
      setLoadingAI(true);
      const prompt = `Analyze the performance of the ${currentHeuristic} heuristic and provide optimization recommendations.`;
      
      const result = await getAIInsights(prompt, {
        heuristic: currentHeuristic,
        schedule_size: currentSchedule.length,
      });
      
      setAiInsights(result.insights);
      enqueueSnackbar('AI insights generated!', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoadingAI(false);
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
          {!currentHeuristic ? (
            <Alert severity="info" sx={{ mb: 3 }}>
              Select a heuristic from the sidebar and click "Compute All Heuristics" to get started.
            </Alert>
          ) : (
            <>
              <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">
                  Active Heuristic: <strong>{currentHeuristic}</strong>
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={loadingAI ? <CircularProgress size={16} /> : <AIIcon />}
                  onClick={handleGetAIInsights}
                  disabled={loadingAI}
                >
                  {loadingAI ? 'Analyzing...' : 'Get AI Insights'}
                </Button>
              </Box>

              {aiInsights && (
                <AIInsightsPanel insights={aiInsights} onClose={() => setAiInsights(null)} />
              )}

              <KPICards />

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
