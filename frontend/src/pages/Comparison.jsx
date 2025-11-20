// src/pages/Comparison.jsx
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Box,
  Button,
  Alert,
} from '@mui/material';
import {
  EmojiEvents as TrophyIcon,
  Refresh as RefreshIcon,
  Psychology as AIIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
import { getMetricsComparison, getAIInsights } from '../services/api';
import AIInsightsPanel from '../components/AIInsightsPanel';

function Comparison() {
  const { enqueueSnackbar } = useSnackbar();
  const { metrics, setMetrics } = useSchedulerStore();
  const [loading, setLoading] = useState(false);
  const [aiInsights, setAiInsights] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const result = await getMetricsComparison();
      
      // Convert array to object with heuristic names as keys
      if (result.metrics && Array.isArray(result.metrics)) {
        const metricsObj = {};
        result.metrics.forEach((m) => {
          if (m.Heuristic) {
            metricsObj[m.Heuristic] = { metrics: m };
          }
        });
        setMetrics(metricsObj);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      // Expected if no metrics computed yet
    } finally {
      setLoading(false);
    }
  };

  const handleGetAIInsights = async () => {
    try {
      setLoadingAI(true);
      const metricsData = Object.values(metrics).map((m) => m.metrics);
      
      const prompt = `Analyze the heuristic performance metrics and identify:
- Which heuristic achieves the best overall performance and why
- Key trade-offs between makespan, tardiness, total cost, and machine utilization
- Specific metric values that support your recommendation
- Any notable strengths or weaknesses of each approach`;
      
      const result = await getAIInsights(prompt, { comparison_data: metricsData });
      setAiInsights(result.insights);
      enqueueSnackbar('AI comparison insights generated!', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoadingAI(false);
    }
  };

  const metricsArray = Object.values(metrics).map((m) => m.metrics || {});

  if (metricsArray.length === 0) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          Heuristic Comparison
        </Typography>
        <Alert severity="info">
          No metrics available. Please load data and compute heuristics from the sidebar.
        </Alert>
      </Container>
    );
  }

  // Find best heuristic for each metric
  const findBest = (metricKey, minimize = true) => {
    let bestValue = minimize ? Infinity : -Infinity;
    let bestHeuristic = '';

    metricsArray.forEach((m) => {
      const value = m[metricKey] || 0;
      if (minimize ? value < bestValue : value > bestValue) {
        bestValue = value;
        bestHeuristic = m.Heuristic;
      }
    });

    return bestHeuristic;
  };

  const bestMakespan = findBest('Makespan_Days', true);
  const bestTardiness = findBest('Total_Tardiness_Days', true);
  const bestCost = findBest('Total_Cost_$', true);
  const bestOnTime = findBest('On_Time_%', false);
  const bestUtilization = findBest('Machine_Utilization_%', false);

  // Overall recommendation (weighted scoring)
  const calculateOverallScore = (m) => {
    return (
      (1 / (m.Makespan_Days || 1)) * 0.25 +
      (1 / (m.Total_Tardiness_Days + 1)) * 0.25 +
      (m['On_Time_%'] || 0) * 0.25 +
      (m['Machine_Utilization_%'] || 0) * 0.25
    );
  };

  let recommendedHeuristic = '';
  let bestScore = -Infinity;
  metricsArray.forEach((m) => {
    const score = calculateOverallScore(m);
    if (score > bestScore) {
      bestScore = score;
      recommendedHeuristic = m.Heuristic;
    }
  });

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h1" gutterBottom>
            Heuristic Comparison
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Compare performance across all scheduling algorithms
          </Typography>
        </Box>
        <Box>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchMetrics}
            disabled={loading}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AIIcon />}
            onClick={handleGetAIInsights}
            disabled={loadingAI}
          >
            {loadingAI ? 'Analyzing...' : 'AI Analysis'}
          </Button>
        </Box>
      </Box>

      {aiInsights && (
        <AIInsightsPanel insights={aiInsights} onClose={() => setAiInsights(null)} />
      )}

      {recommendedHeuristic && (
        <Alert
          severity="success"
          icon={<TrophyIcon />}
          sx={{ mb: 3, fontSize: '1.1rem' }}
        >
          <strong>Recommended:</strong> {recommendedHeuristic} achieves the best overall
          balance across all metrics.
        </Alert>
      )}

      <Card>
        <CardContent>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow sx={{ backgroundColor: '#f8f9fa' }}>
                  <TableCell><strong>Heuristic</strong></TableCell>
                  <TableCell align="right"><strong>Makespan (Days)</strong></TableCell>
                  <TableCell align="right"><strong>Tardiness (Days)</strong></TableCell>
                  <TableCell align="right"><strong>Late Ops</strong></TableCell>
                  <TableCell align="right"><strong>On-Time %</strong></TableCell>
                  <TableCell align="right"><strong>Utilization %</strong></TableCell>
                  <TableCell align="right"><strong>Total Cost ($)</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {metricsArray.map((metric, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Chip
                        label={metric.Heuristic}
                        color={metric.Heuristic === recommendedHeuristic ? 'success' : 'default'}
                        sx={{ fontWeight: 'bold' }}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Box
                        sx={{
                          fontWeight:
                            metric.Heuristic === bestMakespan ? 'bold' : 'normal',
                          color:
                            metric.Heuristic === bestMakespan ? '#10b981' : 'inherit',
                        }}
                      >
                        {metric.Makespan_Days || 0}
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Box
                        sx={{
                          fontWeight:
                            metric.Heuristic === bestTardiness ? 'bold' : 'normal',
                          color:
                            metric.Heuristic === bestTardiness ? '#10b981' : 'inherit',
                        }}
                      >
                        {metric.Total_Tardiness_Days || 0}
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      {metric.Late_Operations || 0} / {metric.Total_Operations || 0}
                    </TableCell>
                    <TableCell align="right">
                      <Box
                        sx={{
                          fontWeight:
                            metric.Heuristic === bestOnTime ? 'bold' : 'normal',
                          color:
                            metric.Heuristic === bestOnTime ? '#10b981' : 'inherit',
                        }}
                      >
                        {metric['On_Time_%'] || 0}%
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Box
                        sx={{
                          fontWeight:
                            metric.Heuristic === bestUtilization ? 'bold' : 'normal',
                          color:
                            metric.Heuristic === bestUtilization ? '#10b981' : 'inherit',
                        }}
                      >
                        {metric['Machine_Utilization_%'] || 0}%
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Box
                        sx={{
                          fontWeight:
                            metric.Heuristic === bestCost ? 'bold' : 'normal',
                          color:
                            metric.Heuristic === bestCost ? '#10b981' : 'inherit',
                        }}
                      >
                        ${(metric['Total_Cost_$'] || 0).toFixed(2)}
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Box sx={{ mt: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
            <Typography variant="caption" color="text.secondary">
              <strong>Note:</strong> Values highlighted in <span style={{ color: '#10b981', fontWeight: 'bold' }}>green</span> represent the best performance for that specific metric.
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
}

export default Comparison;
