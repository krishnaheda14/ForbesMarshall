// src/components/ComputeControls.jsx
import React, { useState, useEffect } from 'react';
import { 
  Button, 
  Box, 
  Typography, 
  CircularProgress, 
  LinearProgress,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Collapse
} from '@mui/material';
import { 
  Calculate as ComputeIcon, 
  CheckCircle as ApplyIcon,
  CheckCircle as DoneIcon,
  Pending as PendingIcon,
  PlayArrow as CurrentIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
import { computeAllHeuristics, applyHeuristic } from '../services/api';

const HEURISTICS = [
  { name: 'SPT', label: 'Shortest Processing Time' },
  { name: 'EDD', label: 'Earliest Due Date' },
  { name: 'CR', label: 'Critical Ratio' },
  { name: 'PRIORITY', label: 'Priority-Based' },
];

function ComputeControls() {
  const { enqueueSnackbar } = useSnackbar();
  const { currentHeuristic, setMetrics, setSchedules, setLoading, loading } = useSchedulerStore();
  const [computeProgress, setComputeProgress] = useState({
    active: false,
    current: null,
    completed: [],
    total: HEURISTICS.length,
    startTime: null
  });
  const [elapsedTime, setElapsedTime] = useState(0);

  // Timer to continuously update elapsed time
  useEffect(() => {
    let interval;
    if (computeProgress.active && computeProgress.startTime) {
      interval = setInterval(() => {
        setElapsedTime(((Date.now() - computeProgress.startTime) / 1000).toFixed(1));
      }, 100); // Update every 100ms for smooth counting
    } else {
      setElapsedTime(0);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [computeProgress.active, computeProgress.startTime]);

  const handleComputeAll = async () => {
    try {
      setLoading(true);
      setComputeProgress({
        active: true,
        current: 'Initializing...',
        completed: [],
        total: HEURISTICS.length,
        startTime: Date.now()
      });
      
      enqueueSnackbar('Starting computation of all heuristics...', { variant: 'info' });
      
      const result = await computeAllHeuristics();
      
      setComputeProgress(prev => ({
        ...prev,
        current: 'Finalizing results...',
      }));
      
      // result.results has structure { HEUR: { schedule_size, metrics } }
      if (result && result.results) {
        const metricsObj = {};
        Object.entries(result.results).forEach(([h, val]) => {
          metricsObj[h] = val.metrics || {};
        });
        setMetrics(metricsObj);
      }
      
      const finalElapsed = ((Date.now() - computeProgress.startTime) / 1000).toFixed(1);
      
      setComputeProgress({
        active: false,
        current: null,
        completed: HEURISTICS.map(h => h.name),
        total: HEURISTICS.length,
        startTime: null
      });
      
      enqueueSnackbar(`All heuristics computed in ${finalElapsed}s!`, { variant: 'success' });
    } catch (error) {
      setComputeProgress({
        active: false,
        current: null,
        completed: [],
        total: HEURISTICS.length,
        startTime: null
      });
      
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleApply = async () => {
    if (!currentHeuristic) {
      enqueueSnackbar('Please select a heuristic first', { variant: 'warning' });
      return;
    }

    try {
      setLoading(true);
      const result = await applyHeuristic(currentHeuristic);
      // Save schedule & metrics in store so Operations view updates immediately
      const { setCurrentSchedule, addSchedule } = useSchedulerStore.getState();
      if (result && result.schedule) {
        setCurrentSchedule(result.schedule);
      }
      if (result && result.metrics) {
        // store the schedule+metrics under the heuristic
        addSchedule(currentHeuristic, result.schedule || [], result.metrics);
      }

      enqueueSnackbar(`${currentHeuristic} applied successfully!`, {
        variant: 'success',
      });
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="caption" sx={{ mb: 1, display: 'block', opacity: 0.9 }}>
        Compute & Apply
      </Typography>
      
      <Button
        fullWidth
        variant="contained"
        size="small"
        startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <ComputeIcon />}
        onClick={handleComputeAll}
        disabled={loading}
        sx={{
          mb: 1,
          backgroundColor: 'rgba(16, 185, 129, 0.9)',
          '&:hover': {
            backgroundColor: 'rgba(16, 185, 129, 1)',
          },
        }}
      >
        {loading ? 'Computing...' : 'Compute All Heuristics'}
      </Button>

      {/* Progress Display */}
      <Collapse in={computeProgress.active}>
        <Paper 
          elevation={0}
          sx={{ 
            p: 1.5, 
            mb: 1, 
            backgroundColor: 'rgba(255,255,255,0.1)',
            backdropFilter: 'blur(10px)'
          }}
        >
          <Box sx={{ mb: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" sx={{ color: 'white', fontWeight: 600 }}>
                Computing: {computeProgress.current}
              </Typography>
              <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                {computeProgress.completed.length}/{computeProgress.total}
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={(computeProgress.completed.length / computeProgress.total) * 100}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: 'rgba(255,255,255,0.2)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#10b981',
                }
              }}
            />
          </Box>
          
          <List dense sx={{ py: 0 }}>
            {HEURISTICS.map((h) => {
              const isCompleted = computeProgress.completed.includes(h.name);
              const isCurrent = computeProgress.current === h.name;
              const isPending = !isCompleted && !isCurrent;
              
              return (
                <ListItem 
                  key={h.name}
                  sx={{ 
                    px: 0.5, 
                    py: 0.25,
                    minHeight: 28,
                    opacity: isPending ? 0.5 : 1
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 28 }}>
                    {isCompleted && <DoneIcon sx={{ fontSize: 16, color: '#10b981' }} />}
                    {isCurrent && <CurrentIcon sx={{ fontSize: 16, color: '#3b82f6' }} />}
                    {isPending && <PendingIcon sx={{ fontSize: 16, color: 'rgba(255,255,255,0.3)' }} />}
                  </ListItemIcon>
                  <ListItemText 
                    primary={h.name}
                    secondary={h.label}
                    primaryTypographyProps={{ 
                      variant: 'caption', 
                      sx: { color: 'white', fontWeight: isCurrent ? 600 : 400 }
                    }}
                    secondaryTypographyProps={{ 
                      variant: 'caption', 
                      sx: { color: 'rgba(255,255,255,0.6)', fontSize: '0.65rem' }
                    }}
                  />
                </ListItem>
              );
            })}
          </List>
          
          {computeProgress.startTime && (
            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)', display: 'block', mt: 0.5, textAlign: 'center' }}>
              Elapsed: {elapsedTime}s
            </Typography>
          )}
        </Paper>
      </Collapse>

      <Button
        fullWidth
        variant="outlined"
        size="small"
        startIcon={<ApplyIcon />}
        onClick={handleApply}
        disabled={!currentHeuristic || loading}
        sx={{
          borderColor: 'rgba(255,255,255,0.5)',
          color: 'white',
          '&:hover': {
            borderColor: 'white',
            backgroundColor: 'rgba(255,255,255,0.1)',
          },
        }}
      >
        Apply Selected
      </Button>
    </Box>
  );
}

export default ComputeControls;
