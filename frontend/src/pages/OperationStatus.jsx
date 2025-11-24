import React, { useEffect, useState, useMemo } from 'react';
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
  TableSortLabel,
  TablePagination,
  CircularProgress
} from '@mui/material';
import { Search as SearchIcon, Download as DownloadIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';
import { getCurrentSchedule } from '../services/api';

function OperationStatus() {
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Pagination State
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  // Sorting State
  const [order, setOrder] = useState('asc');
  const [orderBy, setOrderBy] = useState('Start_Time');

  // 1. Auto-Configure Sort based on selected Heuristic
  useEffect(() => {
    if (currentHeuristic === 'EDD') {
      setOrderBy('Due_Time');
      setOrder('asc');
    } else if (currentHeuristic === 'SPT') {
      setOrderBy('Duration');
      setOrder('asc');
    } else {
      setOrderBy('Start_Time');
      setOrder('asc');
    }
  }, [currentHeuristic]);

  // 2. Fetch Schedule if missing
  useEffect(() => {
    if (currentHeuristic && (!currentSchedule || currentSchedule.length === 0)) {
      fetchSchedule();
    }
  }, [currentHeuristic]);

  const fetchSchedule = async () => {
    try {
      setLoading(true);
      const result = await getCurrentSchedule();
      if (result && result.schedule) {
        setCurrentSchedule(result.schedule);
      }
    } catch (error) {
      console.error("Error fetching schedule");
    } finally {
      setLoading(false);
    }
  };

  const handleRequestSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleExport = () => {
    if (!currentSchedule || currentSchedule.length === 0) return;
    // Export without Proc_Time and Release columns per user request
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
  // --- SAFE FILTERING LOGIC (Fixes hooks ordering crash) ---

  const processedOperations = useMemo(() => {
    const searchLower = (searchTerm || '').toLowerCase();
    
    const filtered = (currentSchedule || []).filter((op) => {
      // We use String() and || '' to ensure we NEVER call toLowerCase on null
      const job = String(op.Job_ID || '').toLowerCase();
      const opId = String(op.Operation_ID || '').toLowerCase();
      const machine = String(op.Machine_ID || '').toLowerCase();
      
      return job.includes(searchLower) || opId.includes(searchLower) || machine.includes(searchLower);
    });

    return filtered.sort((a, b) => {
      // 1. Primary Sort: Priority (Always 1 -> 2 -> 3 -> 4)
      const priorityA = parseInt(a.Priority || 3);
      const priorityB = parseInt(b.Priority || 3);
      
      if (priorityA !== priorityB) {
        return priorityA - priorityB;
      }

      // 2. Secondary Sort
      let aValue = a[orderBy];
      let bValue = b[orderBy];

      if (orderBy === 'Duration') {
        aValue = (a.End_Time || 0) - (a.Start_Time || 0);
        bValue = (b.End_Time || 0) - (b.Start_Time || 0);
      }

      // Handle nulls safely during sort
      if (aValue === undefined || aValue === null) aValue = -Infinity;
      if (bValue === undefined || bValue === null) bValue = -Infinity;

      if (bValue < aValue) return order === 'asc' ? 1 : -1;
      if (bValue > aValue) return order === 'asc' ? -1 : 1;
      return 0;
    });
  }, [currentSchedule, searchTerm, order, orderBy]);

  if (!currentHeuristic) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>📋 Operation Status</Typography>
        <Alert severity="info">No heuristic applied. Please go to Dashboard and Apply a schedule.</Alert>
      </Container>
    );
  }

  // Slice for Pagination
  const visibleRows = processedOperations.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  const getPriorityColor = (priority) => {
    const p = parseInt(priority);
    if (p === 1) return 'error';
    if (p === 2) return 'warning';
    if (p === 3) return 'info';
    return 'default';
  };

  const SortableHeader = ({ id, label, align = 'left' }) => (
    <TableCell align={align} sortDirection={orderBy === id ? order : false}>
      <TableSortLabel
        active={orderBy === id}
        direction={orderBy === id ? order : 'asc'}
        onClick={() => handleRequestSort(id)}
      >
        <strong>{label}</strong>
      </TableSortLabel>
    </TableCell>
  );

  const showProcTime = currentHeuristic === 'SPT';

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h1" gutterBottom>📋 Operation Status</Typography>
          <Typography variant="body1" color="text.secondary">
            Viewing <strong>{processedOperations.length}</strong> operations.
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
          <Button variant="contained" startIcon={<DownloadIcon />} onClick={handleExport}>Export CSV</Button>
        </Box>
      </Box>

      <Card>
        <CardContent>
          <TextField
            fullWidth size="small" placeholder="Search..." value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)} sx={{ mb: 2 }}
            InputProps={{ startAdornment: <InputAdornment position="start"><SearchIcon /></InputAdornment> }}
          />

          {/* CR column removed per request */}

          <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <SortableHeader id="Job_ID" label="Job ID" />
                  <SortableHeader id="Operation_ID" label="Operation ID" />
                  <SortableHeader id="Priority" label="Priority" />
                  <TableCell><strong>Assignment</strong></TableCell>
                  <SortableHeader id="Machine_ID" label="Machine" />
                  {showProcTime && <SortableHeader id="Total_Proc_Min" label="Proc Time (min)" align="right" />}
                  <SortableHeader id="Release_Time_Min" label="Release" align="right" />
                  <SortableHeader id="Start_Time" label="Start" align="right" />
                  <SortableHeader id="End_Time" label="End" align="right" />
                  <SortableHeader id="Duration" label="Duration" align="right" />
                  <SortableHeader id="Due_Time" label="Due Time" align="right" />
                  <SortableHeader id="Tardiness" label="Tardiness" align="right" />
                  <TableCell><strong>Status</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {visibleRows.length > 0 ? (
                  visibleRows.map((op, index) => {
                    const isOutsourced = op.Assignment_Type === 'OUTSOURCE' || op.Machine_ID === 'OUTSOURCE';
                    return (
                      <TableRow key={index} hover>
                        <TableCell>{op.Job_ID}</TableCell>
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
                            size="small" variant={isOutsourced ? 'filled' : 'outlined'}
                          />
                        </TableCell>
                        <TableCell>{op.Machine_ID || '—'}</TableCell>
                        {showProcTime && <TableCell align="right">{
                          (op.Total_Proc_Min ?? op.Proc_Time ?? op.Scheduled_Proc_Time ?? op.proc_time) != null
                            ? Number(op.Total_Proc_Min ?? op.Proc_Time ?? op.Scheduled_Proc_Time ?? op.proc_time).toFixed(0)
                            : '—'
                        }</TableCell>}
                        <TableCell align="right">{
                          (op.Release_Time_Min ?? op.Release_Time ?? op.Release ?? null) != null
                            ? ((op.Release_Time_Min ?? op.Release_Time ?? op.Release).toFixed ? (op.Release_Time_Min ?? op.Release_Time ?? op.Release).toFixed(0) : (op.Release_Time_Min ?? op.Release_Time ?? op.Release))
                            : '—'
                        }</TableCell>
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
                  <TableRow>
                    <TableCell colSpan={12} align="center" sx={{ py: 4 }}>
                      <Typography color="text.secondary">No operations found.</Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          
          <TablePagination
            rowsPerPageOptions={[10, 25, 50, 100]}
            component="div"
            count={processedOperations.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>
    </Container>
  );
}

export default OperationStatus;