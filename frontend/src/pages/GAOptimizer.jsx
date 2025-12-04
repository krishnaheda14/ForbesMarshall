// src/pages/GAOptimizer.jsx
import React, { useState } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Box,
  Button,
  Grid,
  Slider,
  TextField,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  ExpandMore,
  Psychology as AIIcon,
  Timeline as EvolutionIcon,
  Biotech as GeneIcon,
  Assessment as MetricsIcon,
  Lightbulb as ExplainIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { useSnackbar } from 'notistack';
import axios from 'axios';
import useSchedulerStore from '../store/useSchedulerStore';

const API_BASE = 'http://localhost:8001';

function GAOptimizer() {
  const { enqueueSnackbar } = useSnackbar();
  const { setMetrics, setCurrentSchedule } = useSchedulerStore();

  // GA Parameters
  const [populationSize, setPopulationSize] = useState(50);
  const [generations, setGenerations] = useState(100);
  const [mutationRate, setMutationRate] = useState(0.1);
  const [crossoverRate, setCrossoverRate] = useState(0.8);

  // Results
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [evolutionHistory, setEvolutionHistory] = useState([]);
  const [explainability, setExplainability] = useState(null);
  const [debugLogs, setDebugLogs] = useState([]);

  const pushLog = (msg) => {
    const ts = new Date().toLocaleTimeString();
    setDebugLogs((s) => [{ ts, msg }, ...s].slice(0, 200));
    console.debug(`[GA Debug] ${ts} - ${msg}`);
  };

  const runGA = async () => {
    setLoading(true);
    setProgress(0);
    setResults(null);
    setDebugLogs([]);

    pushLog('Starting GA request')

    // Simulate progress: step to 95% gradually
    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + 1, 95));
    }, 500);

    try {
      pushLog('Sending POST /api/schedule/ga')
      const start = Date.now();
      const response = await axios.post(`${API_BASE}/api/schedule/ga`, {
        population_size: populationSize,
        generations: generations,
        mutation_rate: mutationRate,
        crossover_rate: crossoverRate,
        cost_threshold: 0.9,
      }, { timeout: 10 * 60 * 1000 /* 10 minutes */ });

      const took = ((Date.now() - start) / 1000).toFixed(1);
      pushLog(`Response received (status ${response.status}) in ${took}s`)

      clearInterval(progressInterval);
      setProgress(99);

      setResults(response.data);
      setEvolutionHistory(response.data.evolution_history || []);
      setExplainability(response.data.explainability || {});

      // Additional debug info
      const evoLen = (response.data.evolution_history || []).length;
      pushLog(`Evolution history entries: ${evoLen}`);
      if (response.data.explainability && response.data.explainability.best_fitness) {
        pushLog(`Best fitness: ${response.data.explainability.best_fitness.toFixed(6)}`);
      }
      if (evoLen > 0) {
        const lastGen = response.data.evolution_history[evoLen - 1];
        pushLog(`Last generation stats — best: ${lastGen.best_fitness?.toFixed(6) || 'N/A'}, avg: ${lastGen.avg_fitness?.toFixed(6) || 'N/A'}`);
      }

      // Update store
      if (response.data.metrics) {
        setMetrics({ GA: response.data.metrics });
      }

      enqueueSnackbar('🧬 Genetic Algorithm optimization completed!', {
        variant: 'success',
      });
    } catch (error) {
      clearInterval(progressInterval);
      const errMsg = error.response?.data?.detail || error.message || String(error);
      pushLog(`Request failed: ${errMsg}`)
      enqueueSnackbar(`Error: ${errMsg}`, { variant: 'error' });
    } finally {
      setTimeout(() => {
        setLoading(false);
        setProgress(0);
      }, 500);
    }
  };

  const createEvolutionChart = () => {
    if (!evolutionHistory || evolutionHistory.length === 0) return null;

    return {
      data: [
        {
          x: evolutionHistory.map((g) => g.generation),
          y: evolutionHistory.map((g) => g.best_fitness),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Best Fitness',
          line: { color: '#10b981', width: 3 },
          marker: { size: 8, symbol: 'circle' },
        },
        {
          x: evolutionHistory.map((g) => g.generation),
          y: evolutionHistory.map((g) => g.avg_fitness),
          type: 'scatter',
          mode: 'lines',
          name: 'Average Fitness',
          line: { color: '#3b82f6', width: 2, dash: 'dash' },
        },
        {
          x: evolutionHistory.map((g) => g.generation),
          y: evolutionHistory.map((g) => g.worst_fitness),
          type: 'scatter',
          mode: 'lines',
          name: 'Worst Fitness',
          line: { color: '#ef4444', width: 1, dash: 'dot' },
        },
      ],
      layout: {
        title: '🧬 Evolution Progress: Fitness Over Generations',
        xaxis: { title: 'Generation' },
        yaxis: { title: 'Fitness Score (Higher = Better)' },
        height: 400,
        showlegend: true,
      },
    };
  };

  const createMetricsEvolutionChart = () => {
    if (!evolutionHistory || evolutionHistory.length === 0) return null;

    return {
      data: [
        {
          x: evolutionHistory.map((g) => g.generation),
          y: evolutionHistory.map((g) => g.best_makespan),
          type: 'scatter',
          mode: 'lines',
          name: 'Makespan (Days)',
          line: { color: '#8b5cf6', width: 2 },
        },
        {
          x: evolutionHistory.map((g) => g.generation),
          y: evolutionHistory.map((g) => g.best_tardiness),
          type: 'scatter',
          mode: 'lines',
          name: 'Tardiness (Days)',
          line: { color: '#ef4444', width: 2 },
          yaxis: 'y2',
        },
        {
          x: evolutionHistory.map((g) => g.generation),
          y: evolutionHistory.map((g) => g.best_utilization),
          type: 'scatter',
          mode: 'lines',
          name: 'Utilization %',
          line: { color: '#10b981', width: 2 },
          yaxis: 'y3',
        },
      ],
      layout: {
        title: '📊 Performance Metrics Evolution',
        xaxis: { title: 'Generation' },
        yaxis: { title: 'Makespan (Days)', side: 'left' },
        yaxis2: {
          title: 'Tardiness (Days)',
          overlaying: 'y',
          side: 'right',
        },
        yaxis3: {
          title: 'Utilization %',
          overlaying: 'y',
          side: 'right',
          position: 0.85,
        },
        height: 400,
        showlegend: true,
      },
    };
  };

  // Generate unique colors for each job (for Gantt chart)
  const getJobColor = (jobId) => {
    const colors = [
      '#1976d2', '#d32f2f', '#388e3c', '#f57c00', '#7b1fa2',
      '#0097a7', '#c2185b', '#5d4037', '#455a64', '#e64a19',
      '#00796b', '#303f9f', '#c62828', '#6a1b9a', '#0277bd'
    ];
    let hash = 0;
    for (let i = 0; i < jobId.length; i++) {
      hash = jobId.charCodeAt(i) + ((hash << 5) - hash);
    }
    return colors[Math.abs(hash) % colors.length];
  };

  const createGanttChart = () => {
    if (!results || !results.schedule || results.schedule.length === 0) return null;

    // Filter out outsourced operations
    const scheduleData = results.schedule.filter((item) => {
      const assignment = (item.Assignment_Type || '').toString().toUpperCase();
      const machine = (item.Machine_ID || '').toString().toUpperCase();
      return assignment !== 'OUTSOURCE' && machine !== 'OUTSOURCE';
    });

    if (scheduleData.length === 0) return null;

    // Create Gantt traces for each operation
    const ganttData = scheduleData.map((item) => ({
      x: [item.Start_Time, item.End_Time],
      y: [item.Machine_ID, item.Machine_ID],
      type: 'line',
      mode: 'lines',
      line: { width: 20, color: getJobColor(item.Job_ID) },
      name: `${item.Job_ID} - ${item.Operation_ID}`,
      hovertemplate:
        `<b>Machine:</b> ${item.Machine_ID}<br>` +
        `<b>Job:</b> ${item.Job_ID}<br>` +
        `<b>Operation:</b> ${item.Operation_ID}<br>` +
        `<b>Start:</b> ${item.Start_Time} min<br>` +
        `<b>End:</b> ${item.End_Time} min<br>` +
        `<b>Duration:</b> ${item.End_Time - item.Start_Time} min` +
        (item.Priority ? `<br><b>Priority:</b> ${item.Priority}` : '') +
        `<extra></extra>`,
    }));

    // Compute adaptive x-axis tick spacing
    let dtick = 60;
    try {
      const starts = scheduleData.map(t => t.Start_Time).filter(v => typeof v === 'number');
      const ends = scheduleData.map(t => t.End_Time).filter(v => typeof v === 'number');
      if (starts.length && ends.length) {
        const minStart = Math.min(...starts);
        const maxEnd = Math.max(...ends);
        const span = Math.max(1, maxEnd - minStart);
        const candidates = [1,5,10,15,30,60,120,240,480,720,1440, 3000, 5000, 10000, 20000, 50000, 100000];
        const target = Math.ceil(span / 10);
        if (target >= 1000) {
          dtick = candidates.find(c => c >= target) || candidates[candidates.length - 1];
          if (dtick < 1000) dtick = 1000;
        } else {
          dtick = candidates.find(c => c >= target) || candidates[candidates.length - 1];
        }
      }
    } catch (e) {
      console.warn('Failed to compute adaptive dtick for Gantt, defaulting to 60', e);
      dtick = 60;
    }

    return {
      data: ganttData,
      layout: {
        title: '📊 GA Optimized Schedule - Gantt Chart',
        xaxis: {
          title: 'Time (minutes)',
          dtick,
          tickformat: ',.0f',
          showgrid: true,
          zeroline: false,
        },
        yaxis: {
          title: 'Machine',
          autorange: 'reversed',
        },
        height: 500,
        showlegend: false,
        hovermode: 'closest',
      },
    };
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h1" gutterBottom>
        🧬 Genetic Algorithm Optimizer
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Evolution-based optimization that finds near-optimal schedules by
        simulating natural selection
      </Typography>

      {/* What is GA Section */}
      <Alert severity="info" icon={<ExplainIcon />} sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          How Genetic Algorithm Works (Simple Explanation)
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Think of it like breeding better schedules:</strong>
        </Typography>
        <Box component="ul" sx={{ pl: 2, mb: 0 }}>
          <li>
            <Typography variant="body2">
              <strong>Population:</strong> Create {populationSize} different
              random schedules (like {populationSize} different game plans)
            </Typography>
          </li>
          <li>
            <Typography variant="body2">
              <strong>Fitness:</strong> Score each schedule (low cost + low
              tardiness + high utilization = high fitness)
            </Typography>
          </li>
          <li>
            <Typography variant="body2">
              <strong>Selection:</strong> Pick the best schedules to be
              "parents"
            </Typography>
          </li>
          <li>
            <Typography variant="body2">
              <strong>Crossover:</strong> Combine two parents to create
              children (mix good parts from both)
            </Typography>
          </li>
          <li>
            <Typography variant="body2">
              <strong>Mutation:</strong> Randomly change some assignments (try
              new ideas)
            </Typography>
          </li>
          <li>
            <Typography variant="body2">
              <strong>Repeat:</strong> Do this for {generations} generations →
              schedules keep improving!
            </Typography>
          </li>
        </Box>
      </Alert>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            ⚙️ GA Parameters
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Population Size: {populationSize}
              </Typography>
              <Typography variant="caption" color="text.secondary" paragraph>
                More candidates = better exploration, but slower
              </Typography>
              <Slider
                value={populationSize}
                onChange={(e, val) => setPopulationSize(val)}
                min={20}
                max={200}
                step={10}
                marks={[
                  { value: 20, label: '20' },
                  { value: 50, label: '50' },
                  { value: 100, label: '100' },
                  { value: 200, label: '200' },
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Generations: {generations}</Typography>
              <Typography variant="caption" color="text.secondary" paragraph>
                More generations = better solutions, but takes longer
              </Typography>
              <Slider
                value={generations}
                onChange={(e, val) => setGenerations(val)}
                min={50}
                max={500}
                step={50}
                marks={[
                  { value: 50, label: '50' },
                  { value: 100, label: '100' },
                  { value: 200, label: '200' },
                  { value: 500, label: '500' },
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Mutation Rate: {(mutationRate * 100).toFixed(0)}%
              </Typography>
              <Typography variant="caption" color="text.secondary" paragraph>
                Higher = more exploration, lower = faster convergence
              </Typography>
              <Slider
                value={mutationRate}
                onChange={(e, val) => setMutationRate(val)}
                min={0.05}
                max={0.3}
                step={0.05}
                marks={[
                  { value: 0.05, label: '5%' },
                  { value: 0.1, label: '10%' },
                  { value: 0.2, label: '20%' },
                  { value: 0.3, label: '30%' },
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Crossover Rate: {(crossoverRate * 100).toFixed(0)}%
              </Typography>
              <Typography variant="caption" color="text.secondary" paragraph>
                Chance of combining two parents (usually high)
              </Typography>
              <Slider
                value={crossoverRate}
                onChange={(e, val) => setCrossoverRate(val)}
                min={0.5}
                max={1.0}
                step={0.1}
                marks={[
                  { value: 0.5, label: '50%' },
                  { value: 0.7, label: '70%' },
                  { value: 0.8, label: '80%' },
                  { value: 1.0, label: '100%' },
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Button
              variant="contained"
              size="large"
              startIcon={loading ? <CircularProgress size={20} /> : <RunIcon />}
              onClick={runGA}
              disabled={loading}
              sx={{ minWidth: 200 }}
            >
              {loading ? 'Evolving...' : 'Run GA Optimization'}
            </Button>
          </Box>

          {loading && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress variant="determinate" value={progress} />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 1, display: 'block', textAlign: 'center' }}
              >
                Generation {Math.floor((progress / 100) * generations)} /{' '}
                {generations} (estimated)
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {results && (
        <>
          {/* Metrics Comparison */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Final Results
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Card sx={{ bgcolor: '#e3f2fd', textAlign: 'center', p: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Makespan
                    </Typography>
                    <Typography variant="h5">
                      {results.metrics?.Makespan_Days?.toFixed(1) || 0} days
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card sx={{ bgcolor: '#fff3e0', textAlign: 'center', p: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Tardiness
                    </Typography>
                    <Typography variant="h5">
                      {results.metrics?.Total_Tardiness_Days?.toFixed(1) || 0}{' '}
                      days
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card sx={{ bgcolor: '#e8f5e9', textAlign: 'center', p: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Total Cost
                    </Typography>
                    <Typography variant="h5">
                      ${results.metrics?.['Total_Cost_$']?.toFixed(0) || 0}
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card sx={{ bgcolor: '#f3e5f5', textAlign: 'center', p: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Utilization
                    </Typography>
                    <Typography variant="h5">
                      {results.metrics?.['Machine_Utilization_%']?.toFixed(1) ||
                        0}
                      %
                    </Typography>
                  </Card>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Gantt Chart Visualization */}
          {createGanttChart() && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Typography variant="h6">
                    🎯 Final Optimized Schedule - Visual Timeline
                  </Typography>
                </Box>
                <Alert severity="success" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    <strong>Why GA is better:</strong> This Gantt chart shows how the Genetic Algorithm intelligently 
                    optimized machine assignments and operation sequencing. Notice the efficient packing of operations, 
                    minimal idle time, and balanced workload across machines - all achieved through evolutionary optimization 
                    that traditional heuristics cannot match.
                  </Typography>
                </Alert>
                <Plot
                  data={createGanttChart().data}
                  layout={createGanttChart().layout}
                  style={{ width: '100%' }}
                  config={{ responsive: true }}
                />
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>How to read:</strong> Each colored bar represents an operation. Bars of the same color belong to the same job. 
                    Hover over any bar to see detailed information. The horizontal axis shows time in minutes, and the vertical axis shows machines.
                  </Typography>
                </Alert>
              </CardContent>
            </Card>
          )}

          {/* Evolution Charts */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Plot {...createEvolutionChart()} style={{ width: '100%' }} />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Plot
                    {...createMetricsEvolutionChart()}
                    style={{ width: '100%' }}
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Explainability Section */}
          {explainability && (
            <>
              {/* Fitness Breakdown */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <MetricsIcon />
                    <Typography variant="h6">
                      🎯 Fitness Score Breakdown
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Alert severity="success" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>
                        Final Fitness Score:{' '}
                        {explainability.best_fitness?.toFixed(4) || 'N/A'}
                      </strong>
                    </Typography>
                    <Typography variant="caption">
                      Higher is better (combines all objectives into one score)
                    </Typography>
                  </Alert>

                  {explainability.fitness_breakdown && (
                    <TableContainer component={Paper}>
                      <Table size="small">
                        <TableHead>
                          <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                            <TableCell>
                              <strong>Component</strong>
                            </TableCell>
                            <TableCell align="right">
                              <strong>Value</strong>
                            </TableCell>
                            <TableCell align="right">
                              <strong>Contribution to Fitness</strong>
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell>Makespan</TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.makespan_days?.toFixed(
                                1
                              )}{' '}
                              days
                            </TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.makespan_contribution?.toFixed(
                                4
                              )}
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Tardiness</TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.tardiness_days?.toFixed(
                                1
                              )}{' '}
                              days
                            </TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.tardiness_contribution?.toFixed(
                                4
                              )}
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Cost</TableCell>
                            <TableCell align="right">
                              $
                              {explainability.fitness_breakdown.total_cost?.toFixed(
                                0
                              )}
                            </TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.cost_contribution?.toFixed(
                                4
                              )}
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Utilization</TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.utilization_pct?.toFixed(
                                1
                              )}
                              %
                            </TableCell>
                            <TableCell align="right">
                              {explainability.fitness_breakdown.utilization_contribution?.toFixed(
                                4
                              )}
                            </TableCell>
                          </TableRow>
                          {explainability.fitness_breakdown
                            .constraint_violations > 0 && (
                            <TableRow sx={{ bgcolor: '#ffebee' }}>
                              <TableCell>
                                <strong>Constraint Violations</strong>
                              </TableCell>
                              <TableCell align="right">
                                {
                                  explainability.fitness_breakdown
                                    .constraint_violations
                                }
                              </TableCell>
                              <TableCell align="right">
                                -
                                {explainability.fitness_breakdown.violation_penalty?.toFixed(
                                  4
                                )}
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )}
                </AccordionDetails>
              </Accordion>

              {/* Gene Explanations */}
              {explainability.gene_sample_explanations && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <GeneIcon />
                      <Typography variant="h6">
                        🧬 Assignment Decisions (Sample)
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        Each "gene" represents how one operation was assigned to
                        a machine. Here's why the GA made these decisions:
                      </Typography>
                    </Alert>

                    <TableContainer component={Paper}>
                      <Table size="small">
                        <TableHead>
                          <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                            <TableCell>
                              <strong>Operation</strong>
                            </TableCell>
                            <TableCell>
                              <strong>Assignment Decision</strong>
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(
                            explainability.gene_sample_explanations
                          ).map(([opId, explanation]) => (
                            <TableRow key={opId}>
                              <TableCell>
                                <Chip label={opId} size="small" />
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2">
                                  {explanation}
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Evolution Summary */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <EvolutionIcon />
                    <Typography variant="h6">
                      📈 Evolution Summary
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Card sx={{ bgcolor: '#e8f5e9', p: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Fitness Improvement
                        </Typography>
                        <Typography variant="h6">
                          +
                          {explainability.evolution_summary?.improvement?.toFixed(
                            4
                          ) || 0}
                        </Typography>
                        <Typography variant="caption">
                          From first to last generation
                        </Typography>
                      </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Card sx={{ bgcolor: '#e3f2fd', p: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Total Generations
                        </Typography>
                        <Typography variant="h6">
                          {explainability.total_generations || 0}
                        </Typography>
                        <Typography variant="caption">
                          Evolution cycles completed
                        </Typography>
                      </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Card sx={{ bgcolor: '#fff3e0', p: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Final Population
                        </Typography>
                        <Typography variant="h6">
                          {explainability.final_population_size || 0}
                        </Typography>
                        <Typography variant="caption">
                          Candidate solutions explored
                        </Typography>
                      </Card>
                    </Grid>
                  </Grid>

                  {explainability.constraint_violations &&
                    explainability.constraint_violations.length > 0 && (
                      <Alert severity="warning" sx={{ mt: 2 }}>
                        <Typography variant="body2" gutterBottom>
                          <strong>Constraint Violations:</strong>
                        </Typography>
                        <Box component="ul" sx={{ pl: 2, mb: 0 }}>
                          {explainability.constraint_violations.map(
                            (violation, idx) => (
                              <li key={idx}>
                                <Typography variant="caption">
                                  {violation}
                                </Typography>
                              </li>
                            )
                          )}
                        </Box>
                      </Alert>
                    )}
                </AccordionDetails>
              </Accordion>
            </>
          )}

          {/* Debug Log */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="h6">🐞 Debug Log</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {debugLogs.length === 0 && (
                <Typography variant="caption" color="text.secondary">No debug messages yet.</Typography>
              )}
              {debugLogs.map((entry, idx) => (
                <Box key={idx} sx={{ mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">{entry.ts}</Typography>
                  <Typography variant="body2">{entry.msg}</Typography>
                  <Divider sx={{ my: 1 }} />
                </Box>
              ))}
            </AccordionDetails>
          </Accordion>
        </>
      )}

      {/* No Results Yet */}
      {!results && !loading && (
        <Alert severity="info" icon={<AIIcon />}>
          <Typography variant="body1">
            Adjust GA parameters above and click "Run GA Optimization" to find
            the best schedule using evolutionary algorithms.
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            <strong>Recommended settings:</strong> Population=50, Generations=100
            for balanced speed and quality.
          </Typography>
        </Alert>
      )}
    </Container>
  );
}

export default GAOptimizer;
