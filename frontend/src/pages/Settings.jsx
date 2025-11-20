// src/pages/Settings.jsx
import React from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  Box,
  Divider,
  Alert,
} from '@mui/material';
import {
  Delete as ResetIcon,
  Info as InfoIcon,
  BugReport as DebugIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';

function Settings() {
  const { enqueueSnackbar } = useSnackbar();
  const { reset, dataStats, activityLog } = useSchedulerStore();

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset the system? This will clear all data.')) {
      reset();
      enqueueSnackbar('System reset successfully', { variant: 'success' });
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