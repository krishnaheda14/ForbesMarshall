// src/components/MachineryControls.jsx
import React, { useState } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  TextField,
  Button,
  Box,
  Grid,
  InputAdornment,
  Slider,
} from '@mui/material';
import {
  ExpandMore,
  Build as BreakdownIcon,
  PriorityHigh as PriorityIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { simulateBreakdown, updateJobPriority, getCurrentSchedule, computeAllHeuristics, applyHeuristic } from '../services/api';
import useSchedulerStore from '../store/useSchedulerStore';

function MachineryControls() {
  const { enqueueSnackbar } = useSnackbar();
  const { setCurrentSchedule, currentHeuristic } = useSchedulerStore();
  
  // Breakdown state
  const [machineId, setMachineId] = useState('M1');
  const [breakdownStart, setBreakdownStart] = useState(5000);
  const [breakdownDuration, setBreakdownDuration] = useState(100);
  
  // Priority state
  const [jobId, setJobId] = useState('');
  const [priority, setPriority] = useState(2);

  const handleBreakdown = async () => {
    try {
      await simulateBreakdown(machineId, breakdownStart, breakdownDuration);
      
      // Clear all schedules and metrics since breakdown invalidates them
      const { setCurrentSchedule, reset } = useSchedulerStore.getState();
      setCurrentSchedule(null);
      
      enqueueSnackbar('Breakdown simulated! Recomputing heuristics for current view...', {
        variant: 'info',
        autoHideDuration: 3000,
      });

      // Recompute all heuristics and re-apply the currently selected heuristic so the Gantt updates
      try {
        const computeResult = await computeAllHeuristics();
        // update store metrics
        const { setMetrics } = useSchedulerStore.getState();
        if (computeResult && computeResult.results) {
          setMetrics(computeResult.results);
        }

        // Apply previously selected heuristic (or SPT as default)
        const prevHeur = currentHeuristic || 'SPT';
        const applyResult = await applyHeuristic(prevHeur);
        if (applyResult && applyResult.schedule) {
          const { setCurrentSchedule } = useSchedulerStore.getState();
          setCurrentSchedule(applyResult.schedule);
        }

        enqueueSnackbar('Heuristics recomputed and schedule updated with breakdown.', { variant: 'success', autoHideDuration: 4000 });
      } catch (err) {
        console.error('Error recomputing heuristics after breakdown:', err);
        enqueueSnackbar('Breakdown simulated but failed to recompute heuristics. Please recompute manually.', { variant: 'warning' });
      }
      
      console.log('Breakdown simulated, heuristics recomputed and schedule updated');
    } catch (error) {
      console.error('Breakdown simulation error:', error);
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    }
  };

  const handlePriorityUpdate = async () => {
    if (!jobId) {
      enqueueSnackbar('Please enter a Job ID', { variant: 'warning' });
      return;
    }
    
    try {
      // Remember the current heuristic before update
      const previousHeuristic = currentHeuristic || 'PRIORITY';
      
      await updateJobPriority(jobId, priority);
      
      enqueueSnackbar(
        `✅ Priority updated for ${jobId} to ${priority}. Recomputing all heuristics...`,
        { variant: 'info', autoHideDuration: 3000 }
      );
      
      // Automatically recompute all heuristics to show immediate impact
      const result = await computeAllHeuristics();
      const { setMetrics, setCurrentSchedule } = useSchedulerStore.getState();
      setMetrics(result.results);
      
      // Automatically apply the previously selected heuristic (or PRIORITY if none was selected)
      const applyResult = await applyHeuristic(previousHeuristic);
      setCurrentSchedule(applyResult.schedule);
      
      enqueueSnackbar(
        `All heuristics recomputed and ${previousHeuristic} applied! Check Comparison and Operations tabs.`,
        { variant: 'success', autoHideDuration: 5000 }
      );
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    }
  };

  return (
    <Box>
      <Typography variant="caption" sx={{ mb: 1, display: 'block', opacity: 0.9 }}>
        Advanced Controls
      </Typography>

      {/* Machine Breakdown */}
      <Accordion
        sx={{
          backgroundColor: 'rgba(255,255,255,0.1)',
          color: 'white',
          '&:before': { display: 'none' },
          mb: 1,
        }}
      >
        <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <BreakdownIcon sx={{ mr: 1, fontSize: 18 }} />
            <Typography variant="caption">Machine Breakdown</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={1} sx={{ mb: 1.5 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                label="Machine ID"
                value={machineId}
                onChange={(e) => setMachineId(e.target.value)}
                InputLabelProps={{ style: { color: 'rgba(255,255,255,0.7)' } }}
                InputProps={{ style: { color: 'white' } }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                label="Start Time"
                type="number"
                value={breakdownStart}
                onChange={(e) => setBreakdownStart(Number(e.target.value))}
                onBlur={() => {
                  // clamp to valid range
                  const min = 5000; const max = 100000;
                  if (Number.isNaN(breakdownStart) || breakdownStart < min) setBreakdownStart(min);
                  else if (breakdownStart > max) setBreakdownStart(max);
                }}
                helperText="Minutes (5000 - 100000)"
                inputProps={{ min: 5000, max: 100000, step: 1 }}
                InputProps={{
                  endAdornment: <InputAdornment position="end">min</InputAdornment>,
                  style: { color: 'white' }
                }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                label="Duration"
                type="number"
                value={breakdownDuration}
                onChange={(e) => setBreakdownDuration(Number(e.target.value))}
                onBlur={() => {
                  const min = 100; const max = 5000;
                  if (Number.isNaN(breakdownDuration) || breakdownDuration < min) setBreakdownDuration(min);
                  else if (breakdownDuration > max) setBreakdownDuration(max);
                }}
                helperText="Minutes (100 - 5000)"
                inputProps={{ min: 100, max: 5000, step: 1 }}
                InputProps={{
                  endAdornment: <InputAdornment position="end">min</InputAdornment>,
                  style: { color: 'white' }
                }}
              />
            </Grid>
          </Grid>
          <Button
            fullWidth
            size="small"
            variant="contained"
            onClick={handleBreakdown}
            sx={{ backgroundColor: '#ef4444' }}
          >
            Simulate Breakdown
          </Button>
        </AccordionDetails>
      </Accordion>

      {/* Priority Manager */}
      <Accordion
        sx={{
          backgroundColor: 'rgba(255,255,255,0.1)',
          color: 'white',
          '&:before': { display: 'none' },
          mb: 1,
        }}
      >
        <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <PriorityIcon sx={{ mr: 1, fontSize: 18 }} />
            <Typography variant="caption">Job Priority</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <TextField
            fullWidth
            size="small"
            label="Job ID"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
            sx={{ mb: 1.5 }}
            InputLabelProps={{ style: { color: 'rgba(255,255,255,0.7)' } }}
            InputProps={{ style: { color: 'white' } }}
          />
          <Typography variant="caption" gutterBottom>
            Priority: {priority} (1=Highest, 3=Lowest)
          </Typography>
          <Slider
            value={priority}
            onChange={(e, val) => setPriority(val)}
            min={1}
            max={3}
            step={1}
            marks
            sx={{ mb: 1.5, color: 'white' }}
          />
          <Button
            fullWidth
            size="small"
            variant="contained"
            onClick={handlePriorityUpdate}
            sx={{ backgroundColor: '#f59e0b' }}
          >
            Update Priority
          </Button>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}

export default MachineryControls;
