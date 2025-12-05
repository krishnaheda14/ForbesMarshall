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
  CircularProgress,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Divider
} from '@mui/material';
import { Search as SearchIcon, Download as DownloadIcon, Refresh as RefreshIcon, Close as CloseIcon } from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';
import { getCurrentSchedule, computeAllHeuristics, adjustDueDate } from '../services/api';

function OperationStatus() {
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  const [calcOpen, setCalcOpen] = useState(false);
  const [calcOp, setCalcOp] = useState(null);
  const [tardyOpen, setTardyOpen] = useState(false);
  const [tardyOp, setTardyOp] = useState(null);
  const [deltaDays, setDeltaDays] = useState(0);
  const [tardyProcessing, setTardyProcessing] = useState(false);
  
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
      setOrderBy('Proc_Time');
      setOrder('asc');
    } else if (currentHeuristic === 'CR') {
      setOrderBy('Critical_Ratio');
      setOrder('asc');
    } else if (currentHeuristic === 'PRIORITY') {
      setOrderBy('Priority');
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

  // Optimized Filtering & Sorting
  const processedOperations = useMemo(() => {
    const searchLower = (searchTerm || '').toLowerCase();
    
    const filtered = (currentSchedule || []).filter((op) => {
      // 1. ALWAYS HIDE OUTSOURCED OPERATIONS
      // If it's outsourced, we skip it entirely
      if (op.Assignment_Type === 'OUTSOURCE' || op.Machine_ID === 'OUTSOURCE') {
          return false;
      }

      // 2. Search Filter
      const job = String(op.Job_ID || '').toLowerCase();
      const opId = String(op.Operation_ID || '').toLowerCase();
      const machine = String(op.Machine_ID || '').toLowerCase();
      
      return job.includes(searchLower) || opId.includes(searchLower) || machine.includes(searchLower);
    });

    // Sort Logic
    return filtered.sort((a, b) => {
      let aValue = a[orderBy];
      let bValue = b[orderBy];

      // Special handling for specific columns
      if (orderBy === 'Priority') {
         aValue = parseInt(a.Priority || 3);
         bValue = parseInt(b.Priority || 3);
      }

      if (aValue === undefined || aValue === null) aValue = -Infinity;
      if (bValue === undefined || bValue === null) bValue = -Infinity;

      if (bValue < aValue) return order === 'asc' ? 1 : -1;
      if (bValue > aValue) return order === 'asc' ? -1 : 1;
      return 0;
    });
  }, [currentSchedule, searchTerm, order, orderBy]);

  const handleExport = () => {
    if (!processedOperations || processedOperations.length === 0) return;
    
    // Exporting only the visible/filtered operations (No Outsourced)
    const csvHeader = 'Job_ID,Operation_ID,Machine_ID,Priority,Start_Time,End_Time,Proc_Time,Critical_Ratio,Due_Time,Tardiness,Status\n';
    
    const csvRows = processedOperations
      .map((op) => {
        const status = op.Tardiness > 0 ? 'Late' : 'On Time';
        const cr = op.Critical_Ratio !== undefined && op.Critical_Ratio !== null ? op.Critical_Ratio.toFixed(2) : '';
        return `${op.Job_ID},${op.Operation_ID},${op.Machine_ID},${op.Priority},${op.Start_Time},${op.End_Time},${op.Proc_Time},${cr},${op.Due_Time},${op.Tardiness},${status}`;
      })
      .join('\n');

    const blob = new Blob([csvHeader + csvRows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `operations_${currentHeuristic || 'schedule'}.csv`;
    a.click();
  };

  const handleOpenCalc = (op) => {
    setCalcOp(op);
    setCalcOpen(true);
  };

  const handleCloseCalc = () => {
    setCalcOpen(false);
    setCalcOp(null);
  };

  const handleCloseTardy = () => {
    setTardyOpen(false);
    setTardyOp(null);
    setDeltaDays(0);
    setTardyProcessing(false);
  };

  const handleApplyDelta = async () => {
    if (!tardyOp) return;
    try {
      setTardyProcessing(true);
      // Call backend to adjust due date then recompute all heuristics
      await adjustDueDate(tardyOp.Job_ID, Number(deltaDays));
      await computeAllHeuristics();
      // Refresh schedule view
      await fetchSchedule();
      handleCloseTardy();
    } catch (err) {
      console.error('Error applying due date delta', err);
      setTardyProcessing(false);
      // keep dialog open so user can retry
    }
  };

  if (!currentHeuristic) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>Operation Status</Typography>
        <Alert severity="info">No heuristic applied. Please go to Dashboard and Apply a schedule.</Alert>
      </Container>
    );
  }

  const visibleRows = rowsPerPage > 0 
    ? processedOperations.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
    : processedOperations;

  const getPriorityColor = (priority) => {
    const p = parseInt(priority);
    if (p === 1) return 'error';
    if (p === 2) return 'warning';
    if (p === 3) return 'info';
    return 'default';
  };

  // Helper for Sortable Headers
  const SortableHeader = ({ id, label, align = 'left', tooltip = '' }) => (
    <TableCell align={align} sortDirection={orderBy === id ? order : false}>
      <Tooltip title={tooltip} placement="top">
        <TableSortLabel
          active={orderBy === id}
          direction={orderBy === id ? order : 'asc'}
          onClick={() => handleRequestSort(id)}
        >
          <strong>{label}</strong>
        </TableSortLabel>
      </Tooltip>
    </TableCell>
  );

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h1" gutterBottom>Operation Status</Typography>
          <Typography variant="body1" color="text.secondary">
            Viewing <strong>{processedOperations.length}</strong> in-house operations.
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
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

          <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <SortableHeader id="Job_ID" label="Job ID" />
                  <SortableHeader id="Operation_ID" label="Operation ID" />
                  <SortableHeader id="Priority" label="Priority" />
                  <TableCell><strong>Assignment</strong></TableCell>
                  <SortableHeader id="Machine_ID" label="Machine" />
                  
                  <SortableHeader id="Proc_Time" label="Proc Time" align="right" tooltip="Processing Time (mins)" />
                  <SortableHeader id="Critical_Ratio" label="CR" align="right" tooltip="Critical Ratio: (Due - Now) / Work Remaining" />
                  <SortableHeader id="Release_Time" label="Release" align="right" />

                  <SortableHeader id="Start_Time" label="Start" align="right" />
                  <SortableHeader id="End_Time" label="End" align="right" />
                  
                  <SortableHeader id="Due_Time" label="Due Time" align="right" />
                  <SortableHeader id="Tardiness" label="Tardiness" align="right" />
                  <TableCell><strong>Status</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {visibleRows.length > 0 ? (
                  visibleRows.map((op, index) => {
                    // No need to check outsourced here, they are filtered out
                    
                    let crDisplay = '-';
                    let crColor = 'inherit';
                    if (op.Critical_Ratio !== undefined && op.Critical_Ratio !== null) {
                        const crVal = parseFloat(op.Critical_Ratio);
                        crDisplay = crVal.toFixed(2);
                        if (crVal < 0) crColor = '#b71c1c';
                        else if (crVal < 1) crColor = '#d32f2f';
                        else if (crVal < 1.5) crColor = '#f57c00'; 
                        else crColor = '#2e7d32'; 
                    }

                    const releaseTime = (op.Release_Time_Min ?? op.Release_Time ?? op.Release);
                    const releaseDisplay = (releaseTime !== undefined && releaseTime !== null) 
                        ? Number(releaseTime).toFixed(0) 
                        : '-';

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
                            label="In-House"
                            color="primary"
                            size="small" variant="outlined"
                          />
                        </TableCell>
                        <TableCell>{op.Machine_ID || '—'}</TableCell>
                        
                        <TableCell align="right">{op.Proc_Time?.toFixed(0) || '-'}</TableCell>
                        <TableCell align="right" sx={{ color: crColor, fontWeight: 'bold', cursor: 'pointer' }} onClick={() => handleOpenCalc(op)}>
                          {crDisplay}
                        </TableCell>
                        <TableCell align="right">{releaseDisplay}</TableCell>

                        <TableCell align="right">{op.Start_Time?.toFixed(0)}</TableCell>
                        <TableCell align="right">{op.End_Time?.toFixed(0)}</TableCell>
                        
                        <TableCell align="right">{op.Due_Time?.toFixed(0)}</TableCell>
                        <TableCell align="right">
                          <span
                            style={{ color: op.Tardiness > 0 ? '#d32f2f' : '#2e7d32', fontWeight: 'bold', cursor: 'pointer' }}
                            onClick={() => { setTardyOp(op); setTardyOpen(true); setDeltaDays(0); }}
                            title="Click to view tardiness details and adjust due date"
                          >
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
                      <Typography color="text.secondary">No in-house operations found.</Typography>
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
      {/* CR Calculation Dialog */}
      <Dialog open={calcOpen} onClose={handleCloseCalc} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ m: 0, p: 2 }}>
          Critical Ratio Calculation
          <IconButton
            aria-label="close"
            onClick={handleCloseCalc}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <Divider />
        <DialogContent dividers>
          {calcOp ? (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Operation: <strong>{calcOp.Operation_ID}</strong> (Job: {calcOp.Job_ID})
              </Typography>

              {/* Compute input values defensively */}
              {(() => {
                const due = calcOp.Due_Time ?? calcOp.Due_Time_Min ?? calcOp.Due;
                const release = calcOp.Release_Time_Min ?? calcOp.Release_Time ?? calcOp.Release ?? 0;
                const proc = calcOp.Total_Proc_Min ?? calcOp.Proc_Time ?? calcOp.Scheduled_Proc_Time ?? 0;
                const spareProc = proc === 0 ? 1 : proc;
                const cr = ((due || 0) - (release || 0)) / spareProc;

                return (
                  <Box>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      Formula: <code>Critical Ratio = (Due_Time_Min - Release_Time_Min) / Total_Proc_Min</code>
                    </Typography>
                    <Typography variant="body2">Due Time (min): <strong>{Number(due || 0).toFixed(0)}</strong></Typography>
                    <Typography variant="body2">Release Time (min): <strong>{Number(release || 0).toFixed(0)}</strong></Typography>
                    <Typography variant="body2">Work Remaining / Processing Time (min): <strong>{Number(proc || 0).toFixed(0)}</strong></Typography>
                    <Box sx={{ mt: 2, mb: 1 }}>
                      <Typography variant="h6">Computed CR: <span style={{ fontWeight: 'bold' }}>{Number(cr).toFixed(2)}</span></Typography>
                      <Typography variant="caption" color="text.secondary">(If processing time is zero, denominator is treated as 1 to avoid division by zero.)</Typography>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="body2" color="text.secondary">
                      Interpretation: CR &lt; 1 =&gt; operation is critical (due soon relative to remaining work); CR &gt; 1 =&gt; less urgent.
                    </Typography>
                  </Box>
                );
              })()}
            </Box>
          ) : (
            <Typography>No operation selected.</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseCalc} variant="contained">Close</Button>
        </DialogActions>
      </Dialog>
      {/* Tardiness / Due Date Adjustment Dialog */}
      <Dialog open={tardyOpen} onClose={handleCloseTardy} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ m: 0, p: 2 }}>
          Tardiness Details & Adjust Due Date
          <IconButton
            aria-label="close"
            onClick={handleCloseTardy}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <Divider />
        <DialogContent dividers>
          {tardyOp ? (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Operation: <strong>{tardyOp.Operation_ID}</strong> (Job: {tardyOp.Job_ID})
              </Typography>

              {(() => {
                const end = Number(tardyOp.End_Time ?? tardyOp.End_Time_Min ?? tardyOp.End ?? 0);
                const due = Number(tardyOp.Due_Time ?? tardyOp.Due_Time_Min ?? tardyOp.Due ?? 0);
                const tard = Math.max(0, end - due);
                const days = (mins) => (mins / 1440).toFixed(2);

                return (
                  <Box>
                    <Typography variant="body2">Formula: <code>Tardiness = max(0, End_Time - Due_Time_Min)</code></Typography>
                    <Typography variant="body2">End Time (min): <strong>{end.toFixed(0)}</strong></Typography>
                    <Typography variant="body2">Due Time (min): <strong>{due.toFixed(0)}</strong></Typography>
                    <Typography variant="body2">Tardiness (min): <strong style={{ color: tard > 0 ? '#d32f2f' : '#2e7d32' }}>{tard.toFixed(0)}</strong></Typography>
                    <Typography variant="body2">Tardiness (days): <strong style={{ color: tard > 0 ? '#d32f2f' : '#2e7d32' }}>{days(tard)}</strong></Typography>

                    <Divider sx={{ my: 2 }} />

                    <Typography variant="subtitle2" gutterBottom>Adjust Due Date</Typography>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                      Enter days to shift the due date. Use negative values to make the due date earlier, positive to delay.
                    </Typography>
                    <TextField
                      label="Delta Days"
                      type="number"
                      size="small"
                      value={deltaDays}
                      onChange={(e) => setDeltaDays(Number(e.target.value))}
                      InputProps={{ endAdornment: <InputAdornment position="end">days</InputAdornment> }}
                    />
                  </Box>
                );
              })()}
            </Box>
          ) : (
            <Typography>No operation selected.</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseTardy} variant="outlined">Cancel</Button>
          <Button onClick={handleApplyDelta} variant="contained" disabled={tardyProcessing}>
            {tardyProcessing ? 'Applying...' : 'Apply & Recompute'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default OperationStatus;