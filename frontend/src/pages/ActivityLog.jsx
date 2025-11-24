// src/pages/ActivityLog.jsx
import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Box,
  Button,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';

function ActivityLog() {
  const { enqueueSnackbar } = useSnackbar();
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/api/activity-log`);
      if (response.data.status === 'success') {
        setLogs(response.data.log || []);
      }
    } catch (error) {
      enqueueSnackbar(
        `Error: ${error.response?.data?.detail || error.message}`,
        { variant: 'error' }
      );
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (!logs || logs.length === 0) return;
    
    const csvHeader = 'Timestamp,Action,Details\n';
    const csvRows = logs
      .map((log) => {
        const timestamp = log.timestamp || '';
        const action = (log.action || '').replace(/,/g, ';');
        const details = (log.details || '').replace(/,/g, ';');
        return `${timestamp},${action},${details}`;
      })
      .join('\n');
    
    const blob = new Blob([csvHeader + csvRows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `activity_log_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    enqueueSnackbar('Activity log exported successfully', { variant: 'success' });
  };

  const getActionColor = (action) => {
    const actionLower = (action || '').toLowerCase();
    if (actionLower.includes('load')) return 'primary';
    if (actionLower.includes('compute')) return 'info';
    if (actionLower.includes('appli')) return 'success';
    if (actionLower.includes('breakdown') || actionLower.includes('error')) return 'error';
    if (actionLower.includes('update')) return 'warning';
    return 'default';
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <HistoryIcon sx={{ fontSize: 40, color: '#3b82f6' }} />
            <Typography variant="h1" gutterBottom sx={{ mb: 0 }}>
              Activity Log
            </Typography>
          </Box>
          <Typography variant="body1" color="text.secondary">
            Track all actions and changes made to the scheduling system
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={fetchLogs}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={!logs || logs.length === 0}
          >
            Export CSV
          </Button>
        </Box>
      </Box>

      {loading && logs.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : logs.length === 0 ? (
        <Alert severity="info">
          No activity logged yet. Actions like loading data, computing heuristics, or applying schedules will appear here.
        </Alert>
      ) : (
        <Card>
          <CardContent>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Total Events: <strong>{logs.length}</strong>
            </Typography>

            <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Timestamp</strong></TableCell>
                    <TableCell><strong>Action</strong></TableCell>
                    <TableCell><strong>Details</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {[...logs].reverse().map((log, index) => (
                    <TableRow key={index} hover>
                      <TableCell sx={{ whiteSpace: 'nowrap' }}>
                        {log.timestamp || '—'}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={log.action || 'Unknown'}
                          color={getActionColor(log.action)}
                          size="small"
                          sx={{ fontWeight: 'bold' }}
                        />
                      </TableCell>
                      <TableCell>{log.details || '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

export default ActivityLog;
