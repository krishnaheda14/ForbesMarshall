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
  Slider,
} from '@mui/material';
import {
  ExpandMore,
  Build as BreakdownIcon,
  PriorityHigh as PriorityIcon,
  AttachMoney as OutsourceIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { simulateBreakdown, updateJobPriority, updateOutsourcingPolicy, getCurrentSchedule, computeAllHeuristics, applyHeuristic } from '../services/api';
import useSchedulerStore from '../store/useSchedulerStore';

function MachineryControls() {
  const { enqueueSnackbar } = useSnackbar();
  const { setCurrentSchedule, currentHeuristic } = useSchedulerStore();
  
  // Breakdown state
  const [machineId, setMachineId] = useState('M1');
  const [breakdownStart, setBreakdownStart] = useState(1000);
  const [breakdownDuration, setBreakdownDuration] = useState(100);
  
  // Priority state
  const [jobId, setJobId] = useState('');
  const [priority, setPriority] = useState(2);
  
  // Outsourcing state
  const [costThreshold, setCostThreshold] = useState(0.9);

  const handleBreakdown = async () => {
    try {
      await simulateBreakdown(machineId, breakdownStart, breakdownDuration);
      
      // Clear all schedules and metrics since breakdown invalidates them
      const { setCurrentSchedule, reset } = useSchedulerStore.getState();
      setCurrentSchedule(null);
      
      enqueueSnackbar('Breakdown simulated! All schedules cleared. Please recompute heuristics to see rescheduled operations.', {
        variant: 'warning',
        autoHideDuration: 5000,
      });
      
      console.log('Breakdown simulated, schedules cleared');
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

  const handleOutsourcingUpdate = async () => {
    try {
      const result = await updateOutsourcingPolicy(costThreshold);
      
      // Update metrics in store for all recomputed heuristics
      if (result.metrics) {
        Object.entries(result.metrics).forEach(([heur, metrics]) => {
          useSchedulerStore.getState().addSchedule(heur, result.metrics[heur], metrics);
        });
      }
      
      enqueueSnackbar(
        `${result.message} ${result.new_outsourced_count}/${result.total_operations} operations outsourced.`, 
        { variant: 'success' }
      );
      
      // Refresh schedule if a heuristic is active
      if (currentHeuristic) {
        const scheduleResult = await getCurrentSchedule();
        setCurrentSchedule(scheduleResult.schedule);
      }
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
          <TextField
            fullWidth
            size="small"
            label="Machine ID"
            value={machineId}
            onChange={(e) => setMachineId(e.target.value)}
            sx={{ mb: 1.5 }}
            InputLabelProps={{ style: { color: 'rgba(255,255,255,0.7)' } }}
            InputProps={{ style: { color: 'white' } }}
          />
          <Typography variant="caption" gutterBottom>
            Start Time (min): {breakdownStart}
          </Typography>
          <Slider
            value={breakdownStart}
            onChange={(e, val) => setBreakdownStart(val)}
            min={0}
            max={25000}
            step={100}
            sx={{ mb: 1.5, color: 'white' }}
          />
          <Typography variant="caption" gutterBottom>
            Duration (min): {breakdownDuration}
          </Typography>
          <Slider
            value={breakdownDuration}
            onChange={(e, val) => setBreakdownDuration(val)}
            min={0}
            max={500}
            step={10}
            sx={{ mb: 1.5, color: 'white' }}
          />
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
            Priority: {priority} (1=Highest, 4=Lowest)
          </Typography>
          <Slider
            value={priority}
            onChange={(e, val) => setPriority(val)}
            min={1}
            max={4}
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

      {/* Outsourcing Policy */}
      <Accordion
        sx={{
          backgroundColor: 'rgba(255,255,255,0.1)',
          color: 'white',
          '&:before': { display: 'none' },
        }}
      >
        <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <OutsourceIcon sx={{ mr: 1, fontSize: 18 }} />
            <Typography variant="caption">Outsourcing</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="caption" gutterBottom>
            Cost Threshold: {costThreshold.toFixed(2)}
          </Typography>
          <Slider
            value={costThreshold}
            onChange={(e, val) => setCostThreshold(val)}
            min={0.5}
            max={1.5}
            step={0.05}
            sx={{ mb: 1.5, color: 'white' }}
          />
          <Button
            fullWidth
            size="small"
            variant="contained"
            onClick={handleOutsourcingUpdate}
            sx={{ backgroundColor: '#8b5cf6' }}
          >
            Update Policy
          </Button>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}

export default MachineryControls;
