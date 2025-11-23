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
} from '@mui/material';
import { Search as SearchIcon, Download as DownloadIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';
import { getCurrentSchedule } from '../services/api';

function OperationStatus() {
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    if (currentHeuristic && !currentSchedule) {
      fetchSchedule();
    }
  }, [currentHeuristic]);
  
  // Auto-refresh when currentSchedule changes
  useEffect(() => {
    // This will cause re-render when schedule is updated elsewhere
  }, [currentSchedule]);

  const fetchSchedule = async () => {
    try {
      const result = await getCurrentSchedule();
      setCurrentSchedule(result.schedule);
    } catch (error) {
      // Expected
    }
  };

  const handleExport = () => {
    if (!currentSchedule || currentSchedule.length === 0) return;

    const csvHeader = 'Job_ID,Operation_ID,Machine_ID,Start_Time,End_Time,Status\n';
    const csvRows = currentSchedule
      .map((op) => {
        const status = op.Tardiness > 0 ? 'Late' : 'On Time';
        return `${op.Job_ID},${op.Operation_ID},${op.Machine_ID},${op.Start_Time},${op.End_Time},${status}`;
      })
      .join('\n');

    const blob = new Blob([csvHeader + csvRows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `operations_${currentHeuristic}.csv`;
    a.click();
  };

  if (!currentHeuristic || !currentSchedule || currentSchedule.length === 0) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          📋 Operation Status
        </Typography>
        <Alert severity="info">
          No operation data available. Please apply a heuristic first.
        </Alert>
      </Container>
    );
  }

  const filteredOperations = currentSchedule.filter(
    (op) =>
      op.Job_ID?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      op.Operation_ID?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      op.Machine_ID?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h1" gutterBottom>
            📋 Operation Status
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Detailed view of all scheduled operations for {currentHeuristic}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchSchedule}
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
            placeholder="Search by Job ID, Operation ID, or Machine ID..."
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
                  <TableCell align="right"><strong>Start (min)</strong></TableCell>
                  <TableCell align="right"><strong>End (min)</strong></TableCell>
                  <TableCell align="right"><strong>Duration (min)</strong></TableCell>
                  <TableCell align="right"><strong>Due Time (min)</strong></TableCell>
                  <TableCell align="right"><strong>Tardiness (min)</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredOperations.map((op, index) => {
                  const getPriorityColor = (priority) => {
                    const p = typeof priority === 'number' ? priority : parseInt(priority) || 2;
                    if (p === 1) return 'error';      // High
                    if (p === 2) return 'warning';    // Medium
                    if (p === 3) return 'default';    // Low
                    return 'default';
                  };
                  const getPriorityLabel = (priority) => {
                    const p = typeof priority === 'number' ? priority : parseInt(priority) || 2;
                    if (p === 1) return 'High';
                    if (p === 2) return 'Medium';
                    if (p === 3) return 'Low';
                    return `Priority ${p}`;
                  };
                  const assignmentType = op.Assignment_Type || (op.Machine_ID ? 'IN_HOUSE' : 'OUTSOURCE');
                  const isOutsourced = assignmentType === 'OUTSOURCE';
                  
                  return (
                  <TableRow key={index} sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
                    <TableCell><strong>{op.Job_ID}</strong></TableCell>
                    <TableCell>{op.Operation_ID}</TableCell>
                    <TableCell>
                      <Chip
                        label={getPriorityLabel(op.Priority)}
                        color={getPriorityColor(op.Priority)}
                        size="small"
                        sx={{ fontWeight: 'bold' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={isOutsourced ? 'Outsourced' : 'In-House'}
                        color={isOutsourced ? 'secondary' : 'primary'}
                        size="small"
                        variant={isOutsourced ? 'filled' : 'outlined'}
                        sx={{ fontWeight: 'bold' }}
                      />
                    </TableCell>
                    <TableCell>{op.Machine_ID || '—'}</TableCell>
                    <TableCell align="right">{op.Start_Time?.toFixed(0)}</TableCell>
                    <TableCell align="right">{op.End_Time?.toFixed(0)}</TableCell>
                    <TableCell align="right">
                      {(op.End_Time - op.Start_Time)?.toFixed(0)}
                    </TableCell>
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
                )})}
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