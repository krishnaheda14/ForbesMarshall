// src/pages/Settings.jsx
import React, { useState } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  Box,
  Divider,
  Alert,
  TextField,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Delete as ResetIcon,
  Info as InfoIcon,
  BugReport as DebugIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
import { addMachine, removeMachine } from '../services/api';

function Settings() {
  const { enqueueSnackbar } = useSnackbar();
  const { reset, dataStats, activityLog, setMetrics, setCurrentSchedule } = useSchedulerStore();
  
  // Add machine state
  const [newMachine, setNewMachine] = useState({
    machine_id: '',
    machine_type: '',
    op_types: '',
    speed_factor: 1.0,
    hourly_rate: 30.0,
    maintenance_cost: 100.0,
    energy_cost_per_hour: 10.0,
    purchase_price: 50000.0,
  });
  
  // Remove machine state
  const [removeMachineId, setRemoveMachineId] = useState('');
  
  const [loading, setLoading] = useState(false);

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset the system? This will clear all data.')) {
      reset();
      enqueueSnackbar('System reset successfully', { variant: 'success' });
    }
  };
  
  const handleAddMachine = async () => {
    if (!newMachine.machine_id || !newMachine.machine_type || !newMachine.op_types) {
      enqueueSnackbar('Please fill in all required fields (Machine ID, Type, and Op Types)', { variant: 'warning' });
      return;
    }
    
    try {
      setLoading(true);
      const result = await addMachine(newMachine);
      
      // Update store with new metrics
      if (result.results) {
        setMetrics(result.results);
      }
      
      enqueueSnackbar(result.message || 'Machine added successfully!', { variant: 'success' });
      
      // Reset form
      setNewMachine({
        machine_id: '',
        machine_type: '',
        op_types: '',
        speed_factor: 1.0,
        hourly_rate: 30.0,
        maintenance_cost: 100.0,
        energy_cost_per_hour: 10.0,
        purchase_price: 50000.0,
      });
      
      // Refresh schedule if available
      if (result.best_heuristic) {
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  const handleRemoveMachine = async () => {
    if (!removeMachineId) {
      enqueueSnackbar('Please enter a Machine ID to remove', { variant: 'warning' });
      return;
    }
    
    if (!window.confirm(`Are you sure you want to remove machine ${removeMachineId}? This will permanently delete it from the system.`)) {
      return;
    }
    
    try {
      setLoading(true);
      const result = await removeMachine(removeMachineId);
      
      // Update store with new metrics
      if (result.results) {
        setMetrics(result.results);
      }
      
      enqueueSnackbar(result.message || 'Machine removed successfully!', { variant: 'success' });
      setRemoveMachineId('');
      
      // Refresh schedule if available
      if (result.best_heuristic) {
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl">
      <Typography variant="h1" gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        System configuration and management
      </Typography>

      {/* System Info */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <InfoIcon sx={{ mr: 1, color: '#3b82f6' }} />
            <Typography variant="h6">System Information</Typography>
          </Box>
          <Divider sx={{ mb: 2 }} />

          <Box sx={{ display: 'grid', gap: 1 }}>
            <Typography variant="body2">
              <strong>Version:</strong> 2.0.0
            </Typography>
            <Typography variant="body2">
              <strong>Backend:</strong> FastAPI (Python)
            </Typography>
            <Typography variant="body2">
              <strong>Frontend:</strong> React + Material-UI
            </Typography>
            {dataStats && (
              <>
                <Typography variant="body2">
                  <strong>Total Operations:</strong> {dataStats.total_operations}
                </Typography>
                <Typography variant="body2">
                  <strong>Total Machines:</strong> {dataStats.total_machines}
                </Typography>
                <Typography variant="body2">
                  <strong>Total Jobs:</strong> {dataStats.total_jobs}
                </Typography>
              </>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Activity Log */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <DebugIcon sx={{ mr: 1, color: '#10b981' }} />
            <Typography variant="h6">Activity Log</Typography>
          </Box>
          <Divider sx={{ mb: 2 }} />

          {activityLog && activityLog.length > 0 ? (
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {activityLog.slice(-10).reverse().map((log, index) => (
                <Alert key={index} severity="info" sx={{ mb: 1 }}>
                  <Typography variant="caption">
                    <strong>{log.timestamp}:</strong> {log.action}
                  </Typography>
                  <Typography variant="caption" display="block" color="text.secondary">
                    {log.details}
                  </Typography>
                </Alert>
              ))}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No activity logged yet.
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Add Machine */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <AddIcon sx={{ mr: 1, color: '#10b981' }} />
            <Typography variant="h6">Add New Machine</Typography>
          </Box>
          <Divider sx={{ mb: 2 }} />

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Machine ID *"
                value={newMachine.machine_id}
                onChange={(e) => setNewMachine({ ...newMachine, machine_id: e.target.value })}
                placeholder="e.g., M10"
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Machine Type *"
                value={newMachine.machine_type}
                onChange={(e) => setNewMachine({ ...newMachine, machine_type: e.target.value })}
                placeholder="e.g., CNC_Lathe"
                size="small"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Operation Types *"
                value={newMachine.op_types}
                onChange={(e) => setNewMachine({ ...newMachine, op_types: e.target.value })}
                placeholder="Comma-separated, e.g., Turning,Facing,Drilling"
                helperText="Enter operation types separated by commas"
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Speed Factor"
                value={newMachine.speed_factor}
                onChange={(e) => setNewMachine({ ...newMachine, speed_factor: parseFloat(e.target.value) })}
                inputProps={{ min: 0.1, max: 10, step: 0.1 }}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Hourly Rate ($)"
                value={newMachine.hourly_rate}
                onChange={(e) => setNewMachine({ ...newMachine, hourly_rate: parseFloat(e.target.value) })}
                inputProps={{ min: 0, step: 1 }}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Maintenance Cost ($)"
                value={newMachine.maintenance_cost}
                onChange={(e) => setNewMachine({ ...newMachine, maintenance_cost: parseFloat(e.target.value) })}
                inputProps={{ min: 0, step: 10 }}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Energy Cost/Hour ($)"
                value={newMachine.energy_cost_per_hour}
                onChange={(e) => setNewMachine({ ...newMachine, energy_cost_per_hour: parseFloat(e.target.value) })}
                inputProps={{ min: 0, step: 1 }}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Purchase Price ($)"
                value={newMachine.purchase_price}
                onChange={(e) => setNewMachine({ ...newMachine, purchase_price: parseFloat(e.target.value) })}
                inputProps={{ min: 0, step: 1000 }}
                size="small"
              />
            </Grid>
          </Grid>

          <Button
            variant="contained"
            color="success"
            startIcon={<AddIcon />}
            onClick={handleAddMachine}
            disabled={loading}
            sx={{ mt: 2 }}
          >
            Add Machine
          </Button>
        </CardContent>
      </Card>

      {/* Remove Machine */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <RemoveIcon sx={{ mr: 1, color: '#ef4444' }} />
            <Typography variant="h6">Remove Machine</Typography>
          </Box>
          <Divider sx={{ mb: 2 }} />

          <Alert severity="warning" sx={{ mb: 2 }}>
            Removing a machine will permanently delete it from the system and reassign all its tasks.
          </Alert>

          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                label="Machine ID to Remove"
                value={removeMachineId}
                onChange={(e) => setRemoveMachineId(e.target.value)}
                placeholder="e.g., M1"
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                fullWidth
                variant="contained"
                color="error"
                startIcon={<RemoveIcon />}
                onClick={handleRemoveMachine}
                disabled={loading || !removeMachineId}
                sx={{ height: '40px' }}
              >
                Remove
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* System Actions */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Actions
          </Typography>
          <Divider sx={{ mb: 2 }} />

          <Alert severity="warning" sx={{ mb: 2 }}>
            Warning: Resetting the system will clear all computed schedules and loaded data.
          </Alert>

          <Button
            variant="contained"
            color="error"
            startIcon={<ResetIcon />}
            onClick={handleReset}
          >
            Reset System
          </Button>
        </CardContent>
      </Card>
    </Container>
  );
}

export default Settings;