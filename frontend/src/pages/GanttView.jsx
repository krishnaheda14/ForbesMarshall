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
} from '@mui/material';
import Plot from 'react-plotly.js';
import useSchedulerStore from '../store/useSchedulerStore';
import { getCurrentSchedule } from '../services/api';

function GanttView() {
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [loading, setLoading] = useState(false);
  const [maintenanceData, setMaintenanceData] = useState([]);

  useEffect(() => {
    if (currentHeuristic) {
      fetchSchedule();
      fetchMaintenanceData();
    }
  }, [currentHeuristic]);

  const fetchSchedule = async () => {
    try {
      setLoading(true);
      const result = await getCurrentSchedule();
      setCurrentSchedule(result.schedule);
    } catch (error) {
      // Expected if no schedule
    } finally {
      setLoading(false);
    }
  };

  const fetchMaintenanceData = async () => {
    try {
      const result = await getMachineData();
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
      
      setMaintenanceData(maintenance);
    } catch (error) {
      console.error('Failed to fetch maintenance data:', error);
    }
  };

  if (!currentHeuristic || !currentSchedule || currentSchedule.length === 0) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          📈 Gantt Chart
        </Typography>
        <Alert severity="info">
          No schedule data available. Please apply a heuristic from the Dashboard.
        </Alert>
      </Container>
    );
  }

  // Prepare Gantt chart data for operations
  const ganttData = currentSchedule.map((item) => ({
    x: [item.Start_Time, item.End_Time],
    y: [item.Machine_ID, item.Machine_ID],
    type: 'line',
    mode: 'lines',
    line: { width: 20, color: '#1976d2' },
    name: `${item.Job_ID} - ${item.Operation_ID}`,
    hovertemplate:
      `<b>Machine:</b> ${item.Machine_ID}<br>` +
      `<b>Job:</b> ${item.Job_ID}<br>` +
      `<b>Operation:</b> ${item.Operation_ID}<br>` +
      `<b>Start:</b> ${item.Start_Time} min<br>` +
      `<b>End:</b> ${item.End_Time} min<br>` +
      `<b>Duration:</b> ${item.End_Time - item.Start_Time} min<extra></extra>`,
  }));

  // Add maintenance windows (breakdowns) to Gantt chart
  const maintenanceTraces = maintenanceData.map((maint) => ({
    x: [maint.start, maint.end],
    y: [maint.machine, maint.machine],
    type: 'line',
    mode: 'lines',
    line: { width: 20, color: '#ef4444', dash: 'dot' },
    name: `Breakdown - ${maint.machine}`,
    hovertemplate:
      `<b>Machine:</b> ${maint.machine}<br>` +
      `<b>Type:</b> Breakdown/Maintenance<br>` +
      `<b>Start:</b> ${maint.start} min<br>` +
      `<b>End:</b> ${maint.end} min<br>` +
      `<b>Duration:</b> ${maint.duration} min<extra></extra>`,
  }));

  const allTraces = [...ganttData, ...maintenanceTraces];

  const layout = {
    title: `${currentHeuristic} Schedule - Gantt Chart`,
    xaxis: {
      title: 'Time (minutes)',
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

  return (
    <Container maxWidth="xl">
      <Typography variant="h1" gutterBottom>
        📈 Gantt Chart
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Visualizing {currentHeuristic} schedule across machines
      </Typography>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Card>
          <CardContent>
            <Plot data={allTraces} layout={layout} style={{ width: '100%' }} />

            <Alert severity="info" sx={{ mt: 2 }}>
              <strong>Tip:</strong> Hover over bars to see operation details. Blue bars are scheduled operations, red dotted bars are breakdowns/maintenance.
            </Alert>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

export default GanttView;