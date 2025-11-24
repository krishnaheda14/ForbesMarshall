import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Box,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Slider,
  Button,
  Stack,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Factory as FactoryIcon,
  CloudUpload as CloudIcon,
  AttachMoney as MoneyIcon,
  Timer as TimerIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import useSchedulerStore from '../store/useSchedulerStore';
// ✅ FIXED IMPORT: Single line for all API functions
import { getDataInfo, updateOutsourcingPolicy, getCurrentSchedule } from '../services/api';

function OutsourcingAnalysis() {
  const { enqueueSnackbar } = useSnackbar();
  const { currentHeuristic, currentSchedule, setCurrentSchedule } = useSchedulerStore();
  const [loading, setLoading] = useState(true);
  const [policyThreshold, setPolicyThreshold] = useState(0.9); // Default 0.9
  const [updatingPolicy, setUpdatingPolicy] = useState(false);

  useEffect(() => {
    fetchOperationsData();
  }, [currentHeuristic, currentSchedule]);

  const fetchOperationsData = async () => {
    try {
      setLoading(false);
    } catch (error) {
      console.error('Failed to load data:', error);
      setLoading(false);
    }
  };

  const handleUpdatePolicy = async () => {
    try {
      setUpdatingPolicy(true);
      
      // 1. Send update to backend
      const result = await updateOutsourcingPolicy(policyThreshold);
      
      if (result.status === 'success') {
        enqueueSnackbar(`Success! Policy Updated.`, { variant: 'success' });
        
        // 2. CRITICAL STEP: Fetch the NEW schedule immediately
        // This gets the fresh list where jobs are now marked as 'OUTSOURCE'
        const scheduleResult = await getCurrentSchedule();
        if (scheduleResult.schedule) {
          setCurrentSchedule(scheduleResult.schedule);
        }
        
        // 3. Refresh operations data counts
        fetchOperationsData();
      }
    } catch (error) {
      console.error("Policy Update Error:", error);
      const msg = error.response?.data?.detail || error.message;
      enqueueSnackbar(`Failed: ${msg}`, { variant: 'error' });
    } finally {
      setUpdatingPolicy(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl">
        <LinearProgress />
      </Container>
    );
  }

  // Check if we have schedule data
  const hasScheduleData = currentSchedule && Array.isArray(currentSchedule) && currentSchedule.length > 0;

  if (!currentHeuristic || !hasScheduleData) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>💰 Outsourcing Cost Analysis</Typography>
        <Alert severity="info">
          No schedule data available. Please compute and apply a heuristic first from the Dashboard.
        </Alert>
      </Container>
    );
  }

  // Analyze outsourcing decisions
  const totalOps = currentSchedule.length;
  // NEW (Fixed)
  // 1. Identify Outsourced (Strict check)
  const outsourcedOps = currentSchedule.filter(op => 
    op.Assignment_Type === 'OUTSOURCE' || op.Machine_ID === 'OUTSOURCE'
  );

  // 2. Identify In-House (Must NOT be outsourced)
  const inHouseOps = currentSchedule.filter(op => 
    op.Assignment_Type !== 'OUTSOURCE' && op.Machine_ID !== 'OUTSOURCE'
  );

  const outsourcedCount = outsourcedOps.length;
  const inHouseCount = inHouseOps.length;
  const outsourcingRate = totalOps > 0 ? (outsourcedCount / totalOps) * 100 : 0;

  // Cost analysis
  const totalOutsourceCost = outsourcedOps.reduce((sum, op) => sum + (op.Outsource_Cost || 0), 0);
  const totalInHouseCost = inHouseOps.reduce((sum, op) => {
    const duration = (op.End_Time - op.Start_Time) || (op.Total_Proc_Min || 0);
    return sum + (duration / 60 * 30); // Assuming $30/hr
  }, 0);
  const totalCost = totalOutsourceCost + totalInHouseCost;

  // Tardiness comparison
  const outsourcedLate = outsourcedOps.filter(op => op.Tardiness > 0).length;
  const inHouseLate = inHouseOps.filter(op => op.Tardiness > 0).length;
  const outsourcedOnTimeRate = outsourcedOps.length > 0 ? ((outsourcedOps.length - outsourcedLate) / outsourcedOps.length) * 100 : 100;
  const inHouseOnTimeRate = inHouseOps.length > 0 ? ((inHouseOps.length - inHouseLate) / inHouseOps.length) * 100 : 100;

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h1" gutterBottom>💰 Outsourcing Cost Analysis</Typography>
        <Typography variant="body1" color="text.secondary">
          Comprehensive analysis of make-or-buy decisions for {currentHeuristic} heuristic
        </Typography>
      </Box>

      {/* --- CONTROL PANEL --- */}
      <Card sx={{ mb: 4, border: '1px solid #e0e0e0', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
        <CardContent>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="h6" gutterBottom>
                Outsourcing Policy Control
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Adjust the cost threshold. If a vendor is cheaper than <strong>{(policyThreshold * 100).toFixed(0)}%</strong> of our internal cost, we outsource.
                <br/>
                <em>Lower % = Harder to outsource (Conservative). Higher % = Easier to outsource (Aggressive).</em>
              </Typography>
              
              <Stack direction="row" spacing={2} alignItems="center">
                <Typography>Conservative (50%)</Typography>
                <Slider
                  value={policyThreshold}
                  min={0.5}
                  max={1.5}
                  step={0.05}
                  onChange={(e, val) => setPolicyThreshold(val)}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
                  sx={{ color: '#764ba2' }}
                />
                <Typography>Aggressive (150%)</Typography>
              </Stack>
            </Grid>
            <Grid item xs={12} md={4} sx={{ textAlign: 'right' }}>
              <Box>
                <Typography variant="h4" color="primary" fontWeight="bold">
                  {(policyThreshold * 100).toFixed(0)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">Current Threshold</Typography>
              </Box>
              <Button 
                variant="contained" 
                color="primary" 
                size="large"
                startIcon={updatingPolicy ? <RefreshIcon /> : <SaveIcon />}
                onClick={handleUpdatePolicy}
                disabled={updatingPolicy}
                sx={{ mt: 2, px: 4 }}
              >
                {updatingPolicy ? 'Updating...' : 'Update Policy'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>Outsourced</Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>{outsourcedCount}</Typography>
                  <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>{outsourcingRate.toFixed(1)}% of total</Typography>
                </Box>
                <CloudIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>In-House</Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>{inHouseCount}</Typography>
                  <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>{(100 - outsourcingRate).toFixed(1)}% of total</Typography>
                </Box>
                <FactoryIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>Outsource Cost</Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>${totalOutsourceCost.toFixed(0)}</Typography>
                  <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                    {totalCost > 0 ? ((totalOutsourceCost / totalCost) * 100).toFixed(1) : 0}% of total
                  </Typography>
                </Box>
                <MoneyIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>In-House Cost</Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>${totalInHouseCost.toFixed(0)}</Typography>
                  <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                    {totalCost > 0 ? ((totalInHouseCost / totalCost) * 100).toFixed(1) : 0}% of total
                  </Typography>
                </Box>
                <MoneyIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Outsourced Operations Table */}
      {outsourcedOps.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>📋 Outsourced Operations Details</Typography>
            <TableContainer component={Paper} sx={{ maxHeight: 400, mt: 2 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Job ID</strong></TableCell>
                    <TableCell><strong>Operation</strong></TableCell>
                    <TableCell><strong>Priority</strong></TableCell>
                    <TableCell align="right"><strong>Outsource Cost</strong></TableCell>
                    <TableCell align="right"><strong>Est. In-House Cost</strong></TableCell>
                    <TableCell align="right"><strong>Savings</strong></TableCell>
                    <TableCell><strong>Status</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {outsourcedOps.map((op, index) => {
                    const estInHouseCost = ((op.End_Time - op.Start_Time) / 60) * 30 || ((op.Total_Proc_Min || 0) / 60 * 30);
                    const savings = estInHouseCost - (op.Outsource_Cost || 0);
                    const savingsPercent = estInHouseCost > 0 ? (savings / estInHouseCost) * 100 : 0;
                    
                    return (
                      <TableRow key={index}>
                        <TableCell>{op.Job_ID}</TableCell>
                        <TableCell>{op.Operation_ID}</TableCell>
                        <TableCell>
                          <Chip 
                            label={op.Priority || 'N/A'} 
                            size="small" 
                            color={op.Priority === 1 ? 'error' : 'default'}
                          />
                        </TableCell>
                        <TableCell align="right">${(op.Outsource_Cost || 0).toFixed(2)}</TableCell>
                        <TableCell align="right">${estInHouseCost.toFixed(2)}</TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                            <Typography variant="body2" color={savings > 0 ? "success.main" : "error.main"} fontWeight="bold">
                              ${savings.toFixed(2)} ({savingsPercent.toFixed(0)}%)
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            icon={op.Tardiness > 0 ? <WarningIcon /> : <CheckIcon />}
                            label={op.Tardiness > 0 ? 'Late' : 'On Time'}
                            color={op.Tardiness > 0 ? 'error' : 'success'}
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

export default OutsourcingAnalysis;