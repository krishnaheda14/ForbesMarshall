// src/pages/Dashboard.jsx
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
} from '@mui/material';
import {
  CloudUpload as LoadIcon,
  Insights as AIIcon,
  CheckCircle as CheckIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  CloudOff as UnloadIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Close as CloseIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
import { loadData, unloadData, getCurrentSchedule, getAIInsights, getDataInfo, addJob, deleteJob } from '../services/api';
import KPICards from '../components/KPICards';
import AIInsightsPanel from '../components/AIInsightsPanel';
import SchedulingAnimation from '../components/SchedulingAnimation';

function Dashboard() {
  const { enqueueSnackbar } = useSnackbar();
  const {
    dataLoaded,
    dataStats,
    setDataLoaded,
    currentHeuristic,
    currentSchedule,
    setCurrentSchedule,
    loading,
    setLoading,
    metrics,
    reset,
  } = useSchedulerStore();

  const [aiInsights, setAiInsights] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);
  const [addJobDialogOpen, setAddJobDialogOpen] = useState(false);
  const [deleteJobDialogOpen, setDeleteJobDialogOpen] = useState(false);
  const [jobToDelete, setJobToDelete] = useState('');
  
  // UPDATED: Default priority is now 3
  const [newJob, setNewJob] = useState({
    job_id: '',
    operations: [
      {
        operation_type: 'MILLING',
        proc_time: 60,
        setup_time: 10,
        transfer_time: 5,
        quantity: 1,
        release_day: 0,
        due_day: 10,
        priority: 3, // <--- Changed default to Number 3
        vendor_ref: 'V1',
        outsource_cost: 0,
      }
    ]
  });

  useEffect(() => {
    checkDataStatus();
  }, []);

  useEffect(() => {
    if (currentHeuristic && !currentSchedule) {
      fetchCurrentSchedule();
    }
  }, [currentHeuristic]);

  const checkDataStatus = async () => {
    try {
      const result = await getDataInfo();
      if (result.operations > 0) {
        setDataLoaded(true, {
          operations: result.operations,
          machines: result.machines,
          jobs: result.jobs
        });
      }
    } catch (error) {
      console.warn("Backend empty, resetting frontend state.");
      if (dataLoaded) {
        setDataLoaded(false, null);
        reset(); 
      }
    }
  };

  const fetchCurrentSchedule = async () => {
    try {
      const result = await getCurrentSchedule();
      if (result.schedule && result.schedule.length > 0) {
        setCurrentSchedule(result.schedule);
      }
    } catch (error) {
      // Expected if no schedule computed yet
    }
  };

  const handleGetAIInsights = async () => {
    if (!currentSchedule || !currentHeuristic) {
      enqueueSnackbar('Please compute and apply a heuristic first', {
        variant: 'warning',
      });
      return;
    }

    try {
      setLoadingAI(true);
      const currentMetrics = metrics[currentHeuristic] || {};
      const prompt = `Analyze the ${currentHeuristic} scheduling heuristic performance and provide insights:\n- Makespan: ${currentMetrics.Makespan_Days || 'N/A'} days\n- Tardiness: ${currentMetrics.Total_Tardiness_Days || 'N/A'} days\n- Utilization: ${currentMetrics['Machine_Utilization_%'] || 'N/A'}%\n- On-Time: ${currentMetrics['On_Time_%'] || 'N/A'}%\n\nProvide: 1) Performance analysis, 2) Bottlenecks, 3) Recommendations`;
      
      const result = await getAIInsights(prompt, {
        heuristic: currentHeuristic,
        metrics: currentMetrics
      });
      
      setAiInsights(result.insights);
      enqueueSnackbar('AI insights generated!', { variant: 'success' });
    } catch (error) {
      console.error('Failed to fetch AI insights:', error);
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoadingAI(false);
    }
  };

  const handleLoadData = async () => {
    try {
      setLoading(true);
      enqueueSnackbar('Loading dataset...', { variant: 'info' });
      const result = await loadData(50);
      setDataLoaded(true, result.stats);
      enqueueSnackbar('Data loaded successfully!', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleUnloadData = async () => {
    try {
      setLoading(true);
      await unloadData();
      reset();
      enqueueSnackbar('Dataset unloaded successfully!', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleAddJob = async () => {
    try {
      setLoading(true);
      await addJob(newJob);
      enqueueSnackbar(`Job ${newJob.job_id} added successfully!`, { variant: 'success' });
      enqueueSnackbar('Please recompute heuristics to see the updated schedule', { variant: 'info' });
      setAddJobDialogOpen(false);
      // Reset form
      setNewJob({
        job_id: '',
        operations: [
          {
            operation_type: 'MILLING',
            proc_time: 60,
            setup_time: 10,
            transfer_time: 5,
            quantity: 1,
            release_day: 0,
            due_day: 10,
            priority: 3, // Reset to 3
            vendor_ref: 'V1',
            outsource_cost: 0,
          }
        ]
      });
      await checkDataStatus();
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteJob = async () => {
    try {
      setLoading(true);
      await deleteJob(jobToDelete);
      enqueueSnackbar(`Job ${jobToDelete} deleted successfully!`, { variant: 'success' });
      enqueueSnackbar('Please recompute heuristics to see the updated schedule', { variant: 'info' });
      setDeleteJobDialogOpen(false);
      setJobToDelete('');
      await checkDataStatus();
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const addOperation = () => {
    setNewJob({
      ...newJob,
      operations: [
        ...newJob.operations,
        {
          operation_type: 'MILLING',
          proc_time: 60,
          setup_time: 10,
          transfer_time: 5,
          quantity: 1,
          release_day: 0,
          due_day: 10,
          priority: 3, // New operations default to 3
          vendor_ref: 'V1',
          outsource_cost: 0,
        }
      ]
    });
  };

  const removeOperation = (index) => {
    const newOps = newJob.operations.filter((_, i) => i !== index);
    setNewJob({ ...newJob, operations: newOps });
  };

  const updateOperation = (index, field, value) => {
    const newOps = [...newJob.operations];
    newOps[index][field] = value;
    setNewJob({ ...newJob, operations: newOps });
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h1" gutterBottom>
            CNC Scheduling Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Advanced production scheduling with AI-powered insights
          </Typography>
        </Box>
        <Button 
          variant="outlined" 
          color="warning" 
          onClick={handleLoadData}
          startIcon={loading ? <CircularProgress size={20} /> : <WarningIcon />}
          disabled={loading}
        >
          Force Reload Data
        </Button>
      </Box>

      {!dataLoaded ? (
        <Card sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Welcome to CNC Scheduler
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            Load your dataset to begin scheduling operations
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <LoadIcon />}
            onClick={handleLoadData}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Load Dataset'}
          </Button>
        </Card>
      ) : (
        <>
          <Card sx={{ mb: 3, bgcolor: 'success.light' }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box display="flex" alignItems="center" gap={2}>
                  <CheckIcon sx={{ fontSize: 40, color: 'success.dark' }} />
                  <Box>
                    <Typography variant="h6" sx={{ color: 'success.dark' }}>
                      Dataset Loaded Successfully
                    </Typography>
                    <Box display="flex" gap={1} mt={1}>
                      <Chip label={`${dataStats?.operations || 0} Operations`} size="small" color="success" />
                      <Chip label={`${dataStats?.machines || 0} Machines`} size="small" color="success" />
                      <Chip label={`${dataStats?.jobs || 0} Jobs`} size="small" color="success" />
                    </Box>
                  </Box>
                </Box>
                <Box display="flex" gap={1}>
                  <Button variant="outlined" color="primary" startIcon={<AddIcon />} onClick={() => setAddJobDialogOpen(true)}>
                    Add Job
                  </Button>
                  <Button variant="outlined" color="error" startIcon={<DeleteIcon />} onClick={() => setDeleteJobDialogOpen(true)}>
                    Delete Job
                  </Button>
                  <Button variant="outlined" color="success" startIcon={<RefreshIcon />} onClick={handleLoadData} disabled={loading}>
                    Reload
                  </Button>
                  <Button variant="outlined" color="warning" startIcon={<UnloadIcon />} onClick={handleUnloadData} disabled={loading}>
                    Unload
                  </Button>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {!currentHeuristic ? (
            <Alert severity="info" sx={{ mb: 3 }}>
              Select a heuristic from the sidebar and click "Compute All Heuristics" to get started.
            </Alert>
          ) : (
            <>
              <Box sx={{ mb: 3 }}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="h6">
                        Active Heuristic: <strong>{currentHeuristic}</strong>
                      </Typography>
                      <Button
                        variant="outlined"
                        onClick={handleGetAIInsights}
                        disabled={loadingAI || !currentSchedule}
                        startIcon={loadingAI ? <CircularProgress size={20} /> : <AIIcon />}
                      >
                        {loadingAI ? 'Analyzing...' : 'Get AI Insights'}
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Box>

              <KPICards />

              {currentSchedule && currentSchedule.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <SchedulingAnimation schedule={currentSchedule} heuristic={currentHeuristic} />
                </Box>
              )}

              {aiInsights && (
                <Box sx={{ mt: 3 }}>
                  <AIInsightsPanel insights={aiInsights} onClose={() => setAiInsights(null)} />
                </Box>
              )}

              <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>Quick Actions</Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={4}>
                          <Button fullWidth variant="outlined" onClick={() => window.location.href = '/comparison'}>
                            View Comparison
                          </Button>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <Button fullWidth variant="outlined" onClick={() => window.location.href = '/gantt'}>
                            View Gantt Chart
                          </Button>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <Button fullWidth variant="outlined" onClick={() => window.location.href = '/operations'}>
                            View Operations
                          </Button>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </>
          )}
        </>
      )}

      {/* Add Job Dialog */}
      <Dialog open={addJobDialogOpen} onClose={() => setAddJobDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            Add New Job
            <IconButton onClick={() => setAddJobDialogOpen(false)}><CloseIcon /></IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth label="Job ID" value={newJob.job_id}
              onChange={(e) => setNewJob({ ...newJob, job_id: e.target.value })}
              sx={{ mb: 3 }} placeholder="e.g., J101"
            />
            
            <Alert severity="info" sx={{ mb: 3 }}>
              <strong>⚙️ Operation Sequence (Precedence Constraints):</strong><br />
              Operations execute <strong>in sequential order</strong>. The scheduler enforces this automatically.
            </Alert>
            
            <Typography variant="h6" gutterBottom>Operations (Executed in Sequence)</Typography>
            
            {newJob.operations.map((op, index) => (
              <Box key={index}>
                <Card sx={{ mb: 2, p: 2, bgcolor: '#f5f5f5', position: 'relative' }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip label={`Seq ${index + 1}`} color="primary" size="small" sx={{ fontWeight: 'bold' }} />
                      <Typography variant="subtitle1">Operation {index + 1}</Typography>
                    </Box>
                    {newJob.operations.length > 1 && (
                      <IconButton onClick={() => removeOperation(index)} color="error" size="small"><DeleteIcon /></IconButton>
                    )}
                  </Box>
                
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Operation Type</InputLabel>
                        <Select
                          value={op.operation_type}
                          onChange={(e) => updateOperation(index, 'operation_type', e.target.value)}
                          label="Operation Type"
                        >
                          <MenuItem value="MILLING">MILLING (M1, M3, M4)</MenuItem>
                          <MenuItem value="TURNING">TURNING (M6, M9)</MenuItem>
                          <MenuItem value="DRILLING">DRILLING (M1, M3, M4)</MenuItem>
                          <MenuItem value="GRINDING">GRINDING (M6, M9)</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField fullWidth size="small" type="number" label="Processing Time (min)" value={op.proc_time} onChange={(e) => updateOperation(index, 'proc_time', Number(e.target.value))} />
                    </Grid>
                    
                    {/* UPDATED PRIORITY SELECTION - NUMBERS ONLY */}
                    <Grid item xs={12} sm={4}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Priority</InputLabel>
                        <Select
                          value={op.priority}
                          onChange={(e) => updateOperation(index, 'priority', e.target.value)}
                          label="Priority"
                        >
                          <MenuItem value={1}>1 (Highest)</MenuItem>
                          <MenuItem value={2}>2 (High)</MenuItem>
                          <MenuItem value={3}>3 (Medium)</MenuItem>
                          <MenuItem value={4}>4 (Low)</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} sm={4}>
                      <TextField fullWidth size="small" type="number" label="Setup Time (min)" value={op.setup_time} onChange={(e) => updateOperation(index, 'setup_time', Number(e.target.value))} />
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <TextField fullWidth size="small" type="number" label="Quantity" value={op.quantity} onChange={(e) => updateOperation(index, 'quantity', Number(e.target.value))} />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField fullWidth size="small" type="number" label="Due Day" value={op.due_day} onChange={(e) => updateOperation(index, 'due_day', Number(e.target.value))} />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField fullWidth size="small" type="number" label="Outsource Cost ($)" value={op.outsource_cost} onChange={(e) => updateOperation(index, 'outsource_cost', Number(e.target.value))} />
                    </Grid>
                  </Grid>
                </Card>
                {index < newJob.operations.length - 1 && (
                  <Box sx={{ textAlign: 'center', my: 1 }}><Typography variant="h6" color="primary">↓</Typography></Box>
                )}
              </Box>
            ))}
            
            <Button fullWidth variant="outlined" startIcon={<AddIcon />} onClick={addOperation} sx={{ mt: 2 }}>Add Another Operation</Button>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddJobDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddJob} variant="contained" disabled={!newJob.job_id || newJob.operations.length === 0 || loading}>Add Job</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteJobDialogOpen} onClose={() => setDeleteJobDialogOpen(false)}>
        <DialogTitle>Delete Job</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Enter Job ID to delete.</Typography>
          <TextField fullWidth label="Job ID" value={jobToDelete} onChange={(e) => setJobToDelete(e.target.value)} placeholder="e.g., J1" />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteJobDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteJob} variant="contained" color="error" disabled={!jobToDelete || loading}>Delete Job</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default Dashboard;