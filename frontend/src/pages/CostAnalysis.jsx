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
import { Refresh as RefreshIcon, AttachMoney as CostIcon } from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { useSnackbar } from 'notistack';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';

function CostAnalysis() {
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [heuristic, setHeuristic] = useState('SPT');
  const [analysisData, setAnalysisData] = useState(null);

  const hourlyRates = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80];

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/api/analysis/hourly-cost`, {
        heuristic: heuristic,
        hourly_rates: hourlyRates
      });
      
      setAnalysisData(response.data);
      enqueueSnackbar('Cost analysis completed!', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar(
        `Error: ${error.response?.data?.detail || error.message}`,
        { variant: 'error' }
      );
    } finally {
      setLoading(false);
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

          <Alert severity="info" sx={{ mt: 2 }}>
            💡 <strong>What Changes with Hourly Rate:</strong> In-house labor cost increases,
            outsourcing becomes more attractive. <strong>What Doesn't Change:</strong> Tardiness,
            utilization, makespan (these depend on scheduling, not cost).
          </Alert>
        </CardContent>
      </Card>

      {analysisData && (
        <>
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
                    Current Rate ($30/hr)
                  </Typography>
                  <Typography variant="h5">
                    {analysisData.current_outsourcing?.toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    ${analysisData.current_cost?.toFixed(0)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#f3e5f5' }}>
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
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#e8f5e9' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Algorithm Used
                  </Typography>
                  <Typography variant="h5">{heuristic}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {analysisData.total_operations} operations
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
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
          </Grid>

          {/* Data Table */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Detailed Results
              </Typography>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#f8f9fa' }}>
                      <TableCell><strong>Hourly Rate</strong></TableCell>
                      <TableCell align="right"><strong>Outsourcing %</strong></TableCell>
                      <TableCell align="right"><strong>In-House Ops</strong></TableCell>
                      <TableCell align="right"><strong>Outsourced Ops</strong></TableCell>
                      <TableCell align="right"><strong>In-House Cost</strong></TableCell>
                      <TableCell align="right"><strong>Outsource Cost</strong></TableCell>
                      <TableCell align="right"><strong>Total Cost</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {analysisData.results.map((row, index) => (
                      <TableRow key={index}>
                        <TableCell>${row.hourly_rate}/hr</TableCell>
                        <TableCell align="right">{row.outsourcing_pct.toFixed(1)}%</TableCell>
                        <TableCell align="right">{row.inhouse_ops}</TableCell>
                        <TableCell align="right">{row.outsourced_ops}</TableCell>
                        <TableCell align="right">${row.inhouse_cost.toFixed(0)}</TableCell>
                        <TableCell align="right">${row.outsource_cost.toFixed(0)}</TableCell>
                        <TableCell align="right">
                          <strong>${row.total_cost.toFixed(0)}</strong>
                        </TableCell>
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
