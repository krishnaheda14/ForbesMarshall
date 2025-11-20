// src/pages/CostAnalysis.jsx
import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Box,
  Button,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material';
import { Refresh as RefreshIcon, AttachMoney as CostIcon, Psychology as AIIcon } from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { useSnackbar } from 'notistack';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';

function CostAnalysis() {
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [heuristic, setHeuristic] = useState('SPT');
  const [analysisData, setAnalysisData] = useState(null);
  const [aiInsights, setAiInsights] = useState('');
  const [loadingProgress, setLoadingProgress] = useState(0);

  const hourlyRates = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80];

  const cleanAIInsights = (text) => {
    if (!text) return '';
    
    let cleaned = text;
    
    // Remove markdown tables (lines with | characters)
    cleaned = cleaned.split('\n')
      .filter(line => !line.trim().startsWith('|') && !line.includes('---|'))
      .join('\n');
    
    // Remove ** bold markers
    cleaned = cleaned.replace(/\*\*/g, '');
    
    // Remove extra blank lines (more than 1 consecutive)
    cleaned = cleaned.replace(/\n\s*\n\s*\n/g, '\n\n');
    
    // Remove leading/trailing whitespace
    cleaned = cleaned.trim();
    
    return cleaned;
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setLoadingProgress(0);
    setAiInsights('');
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setLoadingProgress(prev => Math.min(prev + 10, 90));
      }, 300);

      const response = await axios.post(`${API_BASE}/api/analysis/hourly-cost`, {
        heuristic: heuristic,
        hourly_rates: hourlyRates
      });
      
      clearInterval(progressInterval);
      setLoadingProgress(100);
      
      setAnalysisData(response.data);
      enqueueSnackbar('Cost analysis completed!', { variant: 'success' });

      // Auto-fetch AI insights
      setTimeout(async () => {
        try {
          const aiResponse = await axios.post(`${API_BASE}/api/ai/insights`, {
            prompt: `Analyze the hourly rate vs cost analysis and outsourcing patterns:\n- Identify optimal hourly rate range that balances cost, outsourcing, and tardiness\n- Explain why specific operation types are outsourced (based on savings data)\n- Evaluate if frequent outsourcing of certain jobs indicates capacity gaps or missing capabilities\n- Highlight key inflection points where cost dramatically impacts outcomes\n- Recommend specific hourly rate and whether to invest in expanding in-house capabilities vs continuing outsourcing\n- Suggest improvements to reduce outsourcing dependency if beneficial\n\nData:\n${JSON.stringify(response.data, null, 2)}`,
            context_data: response.data
          });
          
          setAiInsights(cleanAIInsights(aiResponse.data.insights));
        } catch (error) {
          console.error('Failed to fetch AI insights:', error);
        }
      }, 500);
      
    } catch (error) {
      enqueueSnackbar(
        `Error: ${error.response?.data?.detail || error.message}`,
        { variant: 'error' }
      );
    } finally {
      setTimeout(() => {
        setLoading(false);
        setLoadingProgress(0);
      }, 500);
    }
  };

  const createCostChart = () => {
    if (!analysisData || !analysisData.results) return null;

    const results = analysisData.results;

    return {
      data: [
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.inhouse_cost),
          type: 'bar',
          name: 'In-House Cost',
          marker: { color: '#2E86AB' },
        },
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.outsource_cost),
          type: 'bar',
          name: 'Outsource Cost',
          marker: { color: '#A23B72' },
        },
      ],
      layout: {
        title: '💰 Cost Breakdown by Hourly Rate',
        xaxis: { title: 'Hourly Rate ($/hr)' },
        yaxis: { title: 'Cost ($)' },
        barmode: 'stack',
        height: 400,
      },
    };
  };

  const createOutsourcingChart = () => {
    if (!analysisData || !analysisData.results) return null;

    const results = analysisData.results;

    return {
      data: [
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.outsourcing_pct),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Outsourcing %',
          line: { color: '#F18F01', width: 4 },
          marker: { size: 12, symbol: 'diamond' },
        },
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.outsourced_ops),
          type: 'scatter',
          mode: 'lines+markers',
          name: '# Outsourced Ops',
          line: { color: '#C73E1D', width: 2, dash: 'dash' },
          marker: { size: 8 },
          yaxis: 'y2',
        },
      ],
      layout: {
        title: '📦 Outsourcing % vs Hourly Rate',
        xaxis: { title: 'Hourly Rate ($/hr)' },
        yaxis: { title: 'Outsourcing %' },
        yaxis2: {
          title: 'Number of Operations',
          overlaying: 'y',
          side: 'right',
        },
        height: 400,
      },
    };
  };

  const createTardinessChart = () => {
    if (!analysisData || !analysisData.results) return null;

    const results = analysisData.results;

    return {
      data: [
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.tardiness_days),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Total Tardiness (Days)',
          line: { color: '#E63946', width: 4 },
          marker: { size: 12 },
        },
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.late_operations),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Late Operations',
          line: { color: '#FF6B6B', width: 2, dash: 'dot' },
          marker: { size: 8 },
          yaxis: 'y2',
        },
      ],
      layout: {
        title: 'Tardiness vs Hourly Rate (The Trade-off)',
        xaxis: { title: 'Hourly Rate ($/hr)' },
        yaxis: { title: 'Total Tardiness (Days)' },
        yaxis2: {
          title: 'Number of Late Operations',
          overlaying: 'y',
          side: 'right',
        },
        height: 400,
      },
    };
  };

  const createUtilizationChart = () => {
    if (!analysisData || !analysisData.results) return null;

    const results = analysisData.results;

    return {
      data: [
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.utilization_pct),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Machine Utilization %',
          line: { color: '#06D6A0', width: 4 },
          marker: { size: 12, symbol: 'square' },
        },
        {
          x: results.map(r => r.hourly_rate),
          y: results.map(r => r.on_time_pct),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'On-Time Delivery %',
          line: { color: '#118AB2', width: 3, dash: 'dash' },
          marker: { size: 10 },
          yaxis: 'y',
        },
      ],
      layout: {
        title: 'Capacity Utilization & On-Time Performance',
        xaxis: { title: 'Hourly Rate ($/hr)' },
        yaxis: { title: 'Percentage (%)' },
        height: 400,
      },
    };
  };

  return (
    <Container maxWidth="xl">
      <Typography variant="h1" gutterBottom>
        💰 Hourly Rate vs Cost Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Analyze how hourly labor rates affect in-house costs and outsourcing decisions
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Scheduling Algorithm</InputLabel>
                <Select
                  value={heuristic}
                  onChange={(e) => setHeuristic(e.target.value)}
                  label="Scheduling Algorithm"
                >
                  <MenuItem value="SPT">SPT - Shortest Processing Time</MenuItem>
                  <MenuItem value="EDD">EDD - Earliest Due Date</MenuItem>
                  <MenuItem value="CR">CR - Critical Ratio</MenuItem>
                  <MenuItem value="PRIORITY">PRIORITY - Priority Based</MenuItem>
                  <MenuItem value="WEIGHTED">WEIGHTED - Balanced Multi-Objective</MenuItem>
                  <MenuItem value="SLACK">SLACK - Minimum Slack Time</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <CostIcon />}
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? 'Analyzing...' : 'Run Analysis'}
              </Button>
            </Grid>
          </Grid>

          {/* Loading Progress Bar */}
          {loading && (
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Box sx={{ width: '100%', mr: 1 }}>
                  <Box
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      bgcolor: '#e0e0e0',
                      overflow: 'hidden',
                    }}
                  >
                    <Box
                      sx={{
                        height: '100%',
                        borderRadius: 4,
                        bgcolor: '#1976d2',
                        width: `${loadingProgress}%`,
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </Box>
                </Box>
                <Box sx={{ minWidth: 35 }}>
                  <Typography variant="body2" color="text.secondary">
                    {`${loadingProgress}%`}
                  </Typography>
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {loadingProgress < 30 ? 'Initializing analysis...' : 
                 loadingProgress < 60 ? 'Running scheduling simulations...' : 
                 loadingProgress < 90 ? 'Calculating metrics...' : 
                 'Finalizing results...'}
              </Typography>
            </Box>
          )}

          <Alert severity="info" sx={{ mb: 2 }}>
            <strong>The Trade-off:</strong> Lower hourly rates mean less outsourcing (cheaper in-house).
            BUT keeping everything in-house at low rates increases machine utilization → capacity constraints → tardiness → late deliveries.
            Higher rates increase outsourcing, reducing load and improving on-time performance.
          </Alert>
        </CardContent>
      </Card>

      {analysisData && (
        <>
          {/* AI Insights Panel */}
          {aiInsights && (
            <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <AIIcon sx={{ color: 'white', fontSize: 28 }} />
                  <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
                    ✨ AI Insights & Recommendations
                  </Typography>
                </Box>
                <Box sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.95)', 
                  p: 2, 
                  borderRadius: 2,
                  maxHeight: '400px',
                  overflowY: 'auto'
                }}>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.6 }}>
                    {aiInsights}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Trade-off Insight */}
          <Card sx={{ mb: 3, bgcolor: '#fff3e0' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Key Insight: Cost vs Delivery Trade-off
              </Typography>
              <Typography variant="body1">
                {analysisData.trade_off_insight}
              </Typography>
            </CardContent>
          </Card>

          {/* Key Metrics */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#e3f2fd' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Lowest Total Cost
                  </Typography>
                  <Typography variant="h5">
                    ${analysisData.lowest_cost_rate}/hr
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    ${analysisData.lowest_cost?.toFixed(0)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#fff3e0' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Best On-Time Delivery
                  </Typography>
                  <Typography variant="h5">
                    ${analysisData.best_on_time_rate}/hr
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {analysisData.best_on_time_pct?.toFixed(1)}% on-time
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#f3e5f5' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Lowest Tardiness
                  </Typography>
                  <Typography variant="h5">
                    ${analysisData.lowest_tardiness_rate}/hr
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {analysisData.lowest_tardiness?.toFixed(1)} days
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#e8f5e9' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Max Outsourcing
                  </Typography>
                  <Typography variant="h5">
                    ${analysisData.max_outsource_rate}/hr
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {analysisData.max_outsourcing?.toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts - 4 Charts Now */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Plot {...createCostChart()} style={{ width: '100%' }} />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Plot {...createOutsourcingChart()} style={{ width: '100%' }} />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Plot {...createTardinessChart()} style={{ width: '100%' }} />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Plot {...createUtilizationChart()} style={{ width: '100%' }} />
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Outsourcing Analytics Section */}
          {analysisData.outsourcing_analytics && (
            <Card sx={{ mb: 3, border: '2px solid #1976d2' }}>
              <CardContent>
                <Typography variant="h5" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
                  📦 Outsourcing Analysis & Insights
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Understanding which operations are outsourced, why, and what can be improved
                </Typography>

                <Grid container spacing={3}>
                  {/* Root Causes & Issues */}
                  <Grid item xs={12}>
                    <Card sx={{ bgcolor: '#fff3cd', border: '1px solid #ffc107' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <span>⚠️</span> Root Causes of Outsourcing
                        </Typography>
                        {analysisData.outsourcing_analytics.root_causes.length > 0 ? (
                          <Box component="ul" sx={{ pl: 2, mb: 0 }}>
                            {analysisData.outsourcing_analytics.root_causes.map((cause, idx) => (
                              <li key={idx}>
                                <Typography variant="body1" sx={{ mb: 1 }}>
                                  {cause}
                                </Typography>
                              </li>
                            ))}
                          </Box>
                        ) : (
                          <Typography variant="body1">No outsourcing patterns detected</Typography>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Most Outsourced Operation Types */}
                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          🔧 Most Outsourced Operation Types
                        </Typography>
                        {analysisData.outsourcing_analytics.most_outsourced_operation_types.length > 0 ? (
                          <TableContainer>
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell><strong>Operation Type</strong></TableCell>
                                  <TableCell align="right"><strong>Times Outsourced</strong></TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {analysisData.outsourcing_analytics.most_outsourced_operation_types.map((item, idx) => (
                                  <TableRow key={idx} sx={{ backgroundColor: idx === 0 ? '#fff3e0' : 'white' }}>
                                    <TableCell>{item.operation_type}</TableCell>
                                    <TableCell align="right">
                                      <strong>{item.frequency}</strong>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        ) : (
                          <Alert severity="info">No operations outsourced</Alert>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Savings by Operation Type */}
                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          💰 Average Savings by Operation Type
                        </Typography>
                        {analysisData.outsourcing_analytics.avg_savings_by_operation_type.length > 0 ? (
                          <TableContainer>
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell><strong>Operation Type</strong></TableCell>
                                  <TableCell align="right"><strong>Avg Savings</strong></TableCell>
                                  <TableCell align="right"><strong>Total Saved</strong></TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {analysisData.outsourcing_analytics.avg_savings_by_operation_type.map((item, idx) => (
                                  <TableRow key={idx} sx={{ backgroundColor: idx === 0 ? '#e8f5e9' : 'white' }}>
                                    <TableCell>{item.operation_type}</TableCell>
                                    <TableCell align="right" sx={{ color: '#2e7d32', fontWeight: 'bold' }}>
                                      ${item.avg_savings.toFixed(2)}
                                    </TableCell>
                                    <TableCell align="right">
                                      ${item.total_savings.toFixed(2)}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        ) : (
                          <Alert severity="info">No cost savings data available</Alert>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Most Outsourced Jobs */}
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          📋 Jobs Outsourced Most Frequently
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Jobs appearing here across multiple hourly rates may indicate missing in-house capacity or specialized requirements
                        </Typography>
                        {analysisData.outsourcing_analytics.most_outsourced_jobs.length > 0 ? (
                          <TableContainer>
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell><strong>Job ID</strong></TableCell>
                                  <TableCell align="right"><strong>Outsourcing Frequency</strong></TableCell>
                                  <TableCell><strong>Insight</strong></TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {analysisData.outsourcing_analytics.most_outsourced_jobs.slice(0, 10).map((item, idx) => {
                                  const frequencyPct = (item.frequency / hourlyRates.length) * 100;
                                  let insight = '';
                                  if (frequencyPct >= 80) {
                                    insight = '🔴 Consistently outsourced - Consider expanding in-house capacity';
                                  } else if (frequencyPct >= 50) {
                                    insight = '🟡 Frequently outsourced - May need specialized equipment or skills';
                                  } else {
                                    insight = '🟢 Occasionally outsourced - Cost-driven decision';
                                  }
                                  
                                  return (
                                    <TableRow key={idx} sx={{ backgroundColor: idx < 3 ? '#ffebee' : 'white' }}>
                                      <TableCell><strong>{item.job_id}</strong></TableCell>
                                      <TableCell align="right">
                                        {item.frequency} / {hourlyRates.length} rates ({frequencyPct.toFixed(0)}%)
                                      </TableCell>
                                      <TableCell>{insight}</TableCell>
                                    </TableRow>
                                  );
                                })}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        ) : (
                          <Alert severity="success">
                            All operations can be handled in-house at competitive costs
                          </Alert>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Improvement Recommendations */}
                  <Grid item xs={12}>
                    <Card sx={{ bgcolor: '#e8f5e9', border: '1px solid #4caf50' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ color: '#2e7d32', display: 'flex', alignItems: 'center', gap: 1 }}>
                          <span>💡</span> Recommendations for Improvement
                        </Typography>
                        <Box component="ul" sx={{ pl: 2, mb: 0 }}>
                          {analysisData.outsourcing_analytics.most_outsourced_operation_types.length > 0 && (
                            <li>
                              <Typography variant="body1" sx={{ mb: 1 }}>
                                <strong>Invest in {analysisData.outsourcing_analytics.most_outsourced_operation_types[0].operation_type} capabilities:</strong> This operation type is outsourced most frequently. Adding specialized equipment or training could reduce vendor dependency.
                              </Typography>
                            </li>
                          )}
                          {analysisData.outsourcing_analytics.avg_savings_by_operation_type.length > 0 && (
                            <li>
                              <Typography variant="body1" sx={{ mb: 1 }}>
                                <strong>Negotiate vendor contracts for {analysisData.outsourcing_analytics.avg_savings_by_operation_type[0].operation_type}:</strong> Highest savings potential. Lock in rates with long-term contracts to maintain cost advantage.
                              </Typography>
                            </li>
                          )}
                          {analysisData.outsourcing_analytics.most_outsourced_jobs.length > 0 && (
                            <li>
                              <Typography variant="body1" sx={{ mb: 1 }}>
                                <strong>Review job {analysisData.outsourcing_analytics.most_outsourced_jobs[0].job_id} requirements:</strong> Outsourced across {analysisData.outsourcing_analytics.most_outsourced_jobs[0].frequency} different rate scenarios. Consider if in-house production is feasible with process improvements.
                              </Typography>
                            </li>
                          )}
                          <li>
                            <Typography variant="body1" sx={{ mb: 1 }}>
                              <strong>Monitor outsourcing threshold:</strong> Current decision uses 85% threshold (outsource if vendor cost {"<"} 85% of in-house). Adjust this based on quality, lead time, and strategic considerations.
                            </Typography>
                          </li>
                          <li>
                            <Typography variant="body1">
                              <strong>Balance cost vs delivery:</strong> At ${analysisData.best_on_time_rate}/hr, you achieve {analysisData.best_on_time_pct?.toFixed(1)}% on-time delivery. Consider if the extra cost justifies better service levels.
                            </Typography>
                          </li>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Data Table - Enhanced with Scheduling Metrics */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detailed Results with Scheduling Metrics
              </Typography>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#f8f9fa' }}>
                      <TableCell><strong>Rate</strong></TableCell>
                      <TableCell align="right"><strong>Outsource %</strong></TableCell>
                      <TableCell align="right"><strong>Total Cost</strong></TableCell>
                      <TableCell align="right"><strong>Tardiness (Days)</strong></TableCell>
                      <TableCell align="right"><strong>Late Ops</strong></TableCell>
                      <TableCell align="right"><strong>Utilization %</strong></TableCell>
                      <TableCell align="right"><strong>On-Time %</strong></TableCell>
                      <TableCell align="right"><strong>Makespan (Days)</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {analysisData.results.map((row, index) => (
                      <TableRow 
                        key={index}
                        sx={{
                          backgroundColor: 
                            row.hourly_rate === analysisData.lowest_cost_rate ? '#e3f2fd' :
                            row.hourly_rate === analysisData.best_on_time_rate ? '#e8f5e9' :
                            row.hourly_rate === analysisData.lowest_tardiness_rate ? '#f3e5f5' :
                            'white'
                        }}
                      >
                        <TableCell><strong>${row.hourly_rate}/hr</strong></TableCell>
                        <TableCell align="right">{row.outsourcing_pct.toFixed(1)}%</TableCell>
                        <TableCell align="right">${row.total_cost.toFixed(0)}</TableCell>
                        <TableCell align="right" sx={{ color: row.tardiness_days > 0 ? '#e63946' : 'green' }}>
                          {row.tardiness_days.toFixed(1)}
                        </TableCell>
                        <TableCell align="right" sx={{ color: row.late_operations > 0 ? '#e63946' : 'green' }}>
                          {row.late_operations}
                        </TableCell>
                        <TableCell align="right">{row.utilization_pct.toFixed(1)}%</TableCell>
                        <TableCell align="right" sx={{ color: row.on_time_pct >= 90 ? 'green' : '#f18f01' }}>
                          {row.on_time_pct.toFixed(1)}%
                        </TableCell>
                        <TableCell align="right">{row.makespan_days.toFixed(1)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </>
      )}

      {!analysisData && !loading && (
        <Alert severity="info">
          Select a scheduling algorithm and click "Run Analysis" to see how hourly rates affect
          costs and outsourcing decisions.
        </Alert>
      )}
    </Container>
  );
}

export default CostAnalysis;
