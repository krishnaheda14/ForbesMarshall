// src/pages/GanttView.jsx
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Box,
  Alert,
  CircularProgress,
  Button,
  Chip,
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import Plot from 'react-plotly.js';
import useSchedulerStore from '../store/useSchedulerStore';
import { getCurrentSchedule, getMachineData } from '../services/api';

function GanttView() {
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [loading, setLoading] = useState(true); // Start with true to show loading state
  const [maintenanceData, setMaintenanceData] = useState([]);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Listen for breakdown updates to refresh the chart
  useEffect(() => {
    const handleBreakdownUpdate = () => {
      console.log('Breakdown update event received! Refreshing chart...');
      fetchMaintenanceData();
      setRefreshKey(prev => prev + 1);
    };
    
    window.addEventListener('breakdown-updated', handleBreakdownUpdate);
    console.log('Gantt chart: Registered breakdown-updated event listener');
    
    return () => {
      window.removeEventListener('breakdown-updated', handleBreakdownUpdate);
      console.log('Gantt chart: Unregistered breakdown-updated event listener');
    };
  }, []);

  useEffect(() => {
    if (currentHeuristic) {
      fetchSchedule();
      fetchMaintenanceData();
    } else {
      setLoading(false); // No heuristic selected, stop loading
    }
  }, [currentHeuristic]);

  const fetchSchedule = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await getCurrentSchedule();
      setCurrentSchedule(result.schedule);
    } catch (error) {
      console.error('Failed to fetch schedule:', error);
      setError(error.response?.data?.detail || 'Failed to load schedule data');
    } finally {
      setLoading(false);
    }
  };

  const fetchMaintenanceData = async () => {
    try {
      const result = await getMachineData();
      console.log('Machine data fetched:', result);
      
      if (!result.machines || result.machines.length === 0) {
        console.log('No machine data available yet');
        setMaintenanceData([]);
        return;
      }
      
      const maintenance = [];
      
      result.machines.forEach((machine) => {
        if (machine.Maintenance_Window) {
          const windows = Array.isArray(machine.Maintenance_Window) 
            ? machine.Maintenance_Window 
            : [machine.Maintenance_Window];
          
          windows.forEach((window) => {
            if (window && window.start !== undefined && window.end !== undefined) {
              maintenance.push({
                machine: machine.Machine_ID,
                start: window.start,
                end: window.end,
                duration: window.duration || (window.end - window.start)
              });
            }
          });
        }
      });
      
      console.log('Parsed maintenance windows:', maintenance);
      setMaintenanceData(maintenance);
    } catch (error) {
      console.error('Failed to fetch maintenance data:', error);
    }
  };

  // Loading state
  if (loading) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          📈 Gantt Chart
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 8 }}>
          <CircularProgress size={60} sx={{ mb: 3 }} />
          <Typography variant="h6" color="text.secondary">
            {currentHeuristic ? `Loading ${currentHeuristic} schedule...` : 'Loading...'}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Please wait while we fetch the scheduling data
          </Typography>
        </Box>
      </Container>
    );
  }

  // Error state
  if (error) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          Gantt Chart
        </Typography>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Alert severity="info">
          Please compute a heuristic from the Dashboard first.
        </Alert>
      </Container>
    );
  }

  // No data state
  if (!currentHeuristic || !currentSchedule || currentSchedule.length === 0) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          Gantt Chart
        </Typography>
        <Alert severity="info" sx={{ mb: 2 }}>
          No schedule data available. Please follow these steps:
        </Alert>
        <Box sx={{ mt: 2, p: 3, bgcolor: '#f5f5f5', borderRadius: 2 }}>
          <Typography variant="body1" gutterBottom>
            <strong>Step 1:</strong> Load your dataset from the Dashboard
          </Typography>
          <Typography variant="body1" gutterBottom>
            <strong>Step 2:</strong> Click "Compute All Heuristics" or select a specific heuristic
          </Typography>
          <Typography variant="body1">
            <strong>Step 3:</strong> The Gantt chart will automatically display here
          </Typography>
        </Box>
      </Container>
    );
  }

  // Generate unique colors for each job
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

  // Prepare Gantt chart data for operations with colorful jobs
  // Filter out outsourced operations so they do not appear on the Gantt
  const ganttData = (currentSchedule || []).filter((item) => {
    const assignment = (item.Assignment_Type || '').toString().toUpperCase();
    const machine = (item.Machine_ID || '').toString().toUpperCase();
    return assignment !== 'OUTSOURCE' && machine !== 'OUTSOURCE';
  }).map((item) => ({
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

  // Add maintenance windows (breakdowns) to Gantt chart as solid dark bars
  const maintenanceTraces = maintenanceData.map((maint, idx) => ({
    x: [maint.start, maint.end],
    y: [maint.machine, maint.machine],
    type: 'line',
    mode: 'lines',
    line: { 
      width: 25, 
      color: '#1a1a1a', // Dark black/charcoal color
    },
    name: `Breakdown - ${maint.machine}`,
    hovertemplate:
      `<b>⚠️ BREAKDOWN/MAINTENANCE</b><br>` +
      `<b>Machine:</b> ${maint.machine}<br>` +
      `<b>Start:</b> ${maint.start} min<br>` +
      `<b>End:</b> ${maint.end} min<br>` +
      `<b>Duration:</b> ${maint.duration} min<br>` +
      `<b>Status:</b> Machine Unavailable<extra></extra>`,
    showlegend: true,
    legendgroup: 'breakdown',
  }));

  const allTraces = [...ganttData, ...maintenanceTraces];
  // Compute adaptive x-axis tick spacing based on time span (aim for ~8-12 ticks)
  let dtick = 60;
  try {
    const starts = ganttData.map(t => t.x[0]).filter(v => typeof v === 'number');
    const ends = ganttData.map(t => t.x[1]).filter(v => typeof v === 'number');
    const maintStarts = maintenanceData.map(m => m.start).filter(v => typeof v === 'number');
    const maintEnds = maintenanceData.map(m => m.end).filter(v => typeof v === 'number');
    const allStarts = [...starts, ...maintStarts];
    const allEnds = [...ends, ...maintEnds];
    if (allStarts.length && allEnds.length) {
      const minStart = Math.min(...allStarts);
      const maxEnd = Math.max(...allEnds);
      const span = Math.max(1, maxEnd - minStart);
      // Candidate tick sizes (minutes) including rounded thousands for large spans
      const candidates = [1,5,10,15,30,60,120,240,480,720,1440, 3000, 5000, 10000, 20000, 50000, 100000];
      const target = Math.ceil(span / 10);
      // Prefer a candidate that is a round thousand when span is large
      if (target >= 1000) {
        // round up to nearest thousand-like candidate
        dtick = candidates.find(c => c >= target) || candidates[candidates.length - 1];
        // ensure we pick a nice round thousand (e.g., 1000, 2000, 5000)
        if (dtick < 1000) dtick = 1000;
      } else {
        dtick = candidates.find(c => c >= target) || candidates[candidates.length - 1];
      }
    }
  } catch (e) {
    console.warn('Failed to compute adaptive dtick for Gantt, defaulting to 60', e);
    dtick = 60;
  }

  const layout = {
    title: `${currentHeuristic} Schedule - Gantt Chart`,
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
    height: 600,
    showlegend: false,
  };

  const handleRefresh = async () => {
    console.log('Manual refresh triggered');
    setLoading(true);
    await fetchSchedule();
    await fetchMaintenanceData();
    setRefreshKey(prev => prev + 1);
    setLoading(false);
    console.log('Manual refresh completed');
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h1" gutterBottom>
            Gantt Chart
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Visualizing {currentHeuristic} schedule across machines
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {maintenanceData.length > 0 && (
            <Chip 
              label={`${maintenanceData.length} Breakdown${maintenanceData.length > 1 ? 's' : ''}`} 
              color="error" 
              size="small"
            />
          )}
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : !currentSchedule || currentSchedule.length === 0 ? (
        <Alert severity="info">
          No schedule data available. Please compute a heuristic from the Dashboard first.
        </Alert>
      ) : (
        <Card>
          <CardContent>
            {maintenanceData.length > 0 && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                <strong>{maintenanceData.length} Active Breakdown(s):</strong>
                {maintenanceData.map((m, i) => (
                  <div key={i}>
                    • {m.machine}: {m.start}-{m.end} min (Duration: {m.duration} min)
                  </div>
                ))}
              </Alert>
            )}
            
            <Plot 
              key={refreshKey} 
              data={allTraces} 
              layout={layout} 
              style={{ width: '100%' }} 
            />

            <Alert severity="info" sx={{ mt: 2 }}>
              <strong>Legend:</strong> Hover over bars to see operation details. Each job has a unique color. Dark bars indicate machine breakdowns/maintenance windows.
            </Alert>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

export default GanttView;