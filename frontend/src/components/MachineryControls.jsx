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
import { simulateBreakdown, updateJobPriority, updateOutsourcingPolicy, getCurrentSchedule } from '../services/api';
import useSchedulerStore from '../store/useSchedulerStore';

function MachineryControls() {
  const { enqueueSnackbar } = useSnackbar();
  const { setCurrentSchedule, currentHeuristic } = useSchedulerStore();
  
  // Breakdown state
  const [machineId, setMachineId] = useState('M1');
  const [breakdownStart, setBreakdownStart] = useState(5000);
  const [breakdownDuration, setBreakdownDuration] = useState(240);
  
  // Priority state
  const [jobId, setJobId] = useState('');
  const [priority, setPriority] = useState(2);
  
  // Outsourcing state
  const [costThreshold, setCostThreshold] = useState(0.9);

  const handleBreakdown = async () => {
    try {
      await simulateBreakdown(machineId, breakdownStart, breakdownDuration);
      enqueueSnackbar('Breakdown simulated. Recompute heuristics to see impact.', {
        variant: 'success',
      });
    } catch (error) {
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
      await updateJobPriority(jobId, priority);
      enqueueSnackbar(
        `✅ Priority updated for ${jobId} to ${priority}. Check Operation Status tab to verify!`,
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
      enqueueSnackbar(
        `Outsourcing policy updated! ${result.new_outsourced_count}/${result.total_operations} operations outsourced.`, 
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
            min={5000}
            max={20000}
            step={100}
            sx={{ mb: 1.5, color: 'white' }}
          />
          <Typography variant="caption" gutterBottom>
            Duration (min): {breakdownDuration}
          </Typography>
          <Slider
            value={breakdownDuration}
            onChange={(e, val) => setBreakdownDuration(val)}
            min={30}
            max={5000}
            step={30}
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
