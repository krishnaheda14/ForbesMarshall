// src/pages/OperationStatus.jsx
import React, { useEffect, useState } from 'react';
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
  Chip,
  Box,
  Alert,
  TextField,
  InputAdornment,
  Button,
  CircularProgress
} from '@mui/material';
import { Search as SearchIcon, Download as DownloadIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';
import { getCurrentSchedule } from '../services/api';

function OperationStatus() {
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);

  // Initial fetch
  useEffect(() => {
    if (currentHeuristic && (!currentSchedule || currentSchedule.length === 0)) {
      fetchSchedule();
    }
  }, [currentHeuristic]);

  // DEBUG: Print schedule to console to verify data
  useEffect(() => {
    console.log("Current Schedule Data:", currentSchedule);
  }, [currentSchedule]);

  const fetchSchedule = async () => {
    try {
      setLoading(true);
      const result = await getCurrentSchedule();
      if (result && result.schedule) {
        setCurrentSchedule(result.schedule);
      }
    } catch (error) {
      console.error("Error fetching schedule:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (!currentSchedule || currentSchedule.length === 0) return;

    const csvHeader = 'Job_ID,Operation_ID,Machine_ID,Start_Time,End_Time,Status,Priority\n';
    const csvRows = currentSchedule
      .map((op) => {
        const status = op.Tardiness > 0 ? 'Late' : 'On Time';
        return `${op.Job_ID},${op.Operation_ID},${op.Machine_ID},${op.Start_Time},${op.End_Time},${status},${op.Priority}`;
      })
      .join('\n');

    const blob = new Blob([csvHeader + csvRows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `operations_${currentHeuristic || 'schedule'}.csv`;
    a.click();
  };

  // 1. Check if Heuristic is applied
  if (!currentHeuristic) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>📋 Operation Status</Typography>
        <Alert severity="info">
          No heuristic applied. Please go to the <strong>Dashboard</strong>, select a heuristic (e.g., SPT), click <strong>Compute</strong>, and then <strong>Apply</strong>.
        </Alert>
      </Container>
    );
  }

  // 2. Check if Schedule exists
  if (!currentSchedule || currentSchedule.length === 0) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>📋 Operation Status</Typography>
        <Alert severity="warning" action={
          <Button color="inherit" size="small" onClick={fetchSchedule}>Try Refresh</Button>
        }>
          Heuristic is applied ({currentHeuristic}), but no schedule data was found.
          <br />
          <strong>Try clicking "Compute All" and "Apply" again on the Dashboard.</strong>
        </Alert>
      </Container>
    );
  }

  // 3. Robust Filtering (Won't crash on nulls)
  const filteredOperations = currentSchedule.filter((op) => {
    const search = (searchTerm || '').toLowerCase();
    if (!search) return true; // Show all if search is empty

    // Safely convert fields to string before searching
    const job = String(op.Job_ID || '').toLowerCase();
    const operation = String(op.Operation_ID || '').toLowerCase();
    const machine = String(op.Machine_ID || '').toLowerCase();
    
    return job.includes(search) || operation.includes(search) || machine.includes(search);
  });

  // Helper for Priority Colors (1=Red, 2=Orange, 3=Blue, 4=Grey)
  const getPriorityColor = (priority) => {
    const p = parseInt(priority);
    if (p === 1) return 'error';      // High
    if (p === 2) return 'warning';    // Medium-High
    if (p === 3) return 'info';       // Medium-Low
    return 'default';                 // Low
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h1" gutterBottom>
            📋 Operation Status
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Viewing <strong>{currentSchedule.length}</strong> operations for <strong>{currentHeuristic}</strong>
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={fetchSchedule}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
          >
            Export CSV
          </Button>
        </Box>
      </Box>

      <Card>
        <CardContent>
          <TextField
            fullWidth
            size="small"
            placeholder="Search by Job, Operation, or Machine..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            sx={{ mb: 2 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />

          <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Job ID</strong></TableCell>
                  <TableCell><strong>Operation ID</strong></TableCell>
                  <TableCell><strong>Priority</strong></TableCell>
                  <TableCell><strong>Assignment</strong></TableCell>
                  <TableCell><strong>Machine</strong></TableCell>
                  <TableCell align="right"><strong>Start</strong></TableCell>
                  <TableCell align="right"><strong>End</strong></TableCell>
                  <TableCell align="right"><strong>Duration</strong></TableCell>
                  <TableCell align="right"><strong>Due</strong></TableCell>
                  <TableCell align="right"><strong>Tardiness</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredOperations.length > 0 ? (
                  filteredOperations.map((op, index) => {
                    const assignmentType = op.Assignment_Type || (op.Machine_ID === 'OUTSOURCE' ? 'OUTSOURCE' : 'IN_HOUSE');
                    const isOutsourced = assignmentType === 'OUTSOURCE';
                    
                    return (
                      <TableRow key={index} sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
                        <TableCell><strong>{op.Job_ID}</strong></TableCell>
                        <TableCell>{op.Operation_ID}</TableCell>
                        <TableCell>
                          <Chip
                            label={op.Priority}
                            color={getPriorityColor(op.Priority)}
                            size="small"
                            sx={{ fontWeight: 'bold', minWidth: 30 }}
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={isOutsourced ? 'Outsourced' : 'In-House'}
                            color={isOutsourced ? 'secondary' : 'primary'}
                            size="small"
                            variant={isOutsourced ? 'filled' : 'outlined'}
                          />
                        </TableCell>
                        <TableCell>{op.Machine_ID || '—'}</TableCell>
                        <TableCell align="right">{op.Start_Time?.toFixed(0)}</TableCell>
                        <TableCell align="right">{op.End_Time?.toFixed(0)}</TableCell>
                        <TableCell align="right">{(op.End_Time - op.Start_Time)?.toFixed(0)}</TableCell>
                        <TableCell align="right">{op.Due_Time?.toFixed(0)}</TableCell>
                        <TableCell align="right">
                          <span style={{ color: op.Tardiness > 0 ? '#d32f2f' : '#2e7d32', fontWeight: 'bold' }}>
                            {op.Tardiness > 0 ? op.Tardiness.toFixed(0) : '0'}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={op.Tardiness > 0 ? 'Late' : 'On Time'}
                            color={op.Tardiness > 0 ? 'error' : 'success'}
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })
                ) : (
                  // Fallback if filter matches nothing
                  <TableRow>
                    <TableCell colSpan={11} align="center" sx={{ py: 4 }}>
                      <Typography variant="body1" color="text.secondary">
                        No operations found matching "{searchTerm}"
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Showing {filteredOperations.length} of {currentSchedule.length} operations
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
}

export default OperationStatus;