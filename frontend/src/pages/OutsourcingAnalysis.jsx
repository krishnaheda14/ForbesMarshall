// src/pages/OutsourcingAnalysis.jsx
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
} from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';
import { getDataInfo } from '../services/api';

function OutsourcingAnalysis() {
  const { currentHeuristic, currentSchedule } = useSchedulerStore();
  const [opsData, setOpsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchOperationsData();
  }, [currentHeuristic, currentSchedule]); // Re-fetch when schedule changes

  const fetchOperationsData = async () => {
    try {
      const result = await getDataInfo();
      setOpsData(result.operations);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load operations data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl">
        <LinearProgress />
      </Container>
    );
  }

  // Check if we have schedule data from the store
  const hasScheduleData = currentSchedule && Array.isArray(currentSchedule) && currentSchedule.length > 0;

  if (!currentHeuristic || !hasScheduleData) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h1" gutterBottom>
          💰 Outsourcing Cost Analysis
        </Typography>
        <Alert severity="info">
          No schedule data available. Please compute and apply a heuristic first from the Dashboard.
        </Alert>
      </Container>
    );
  }

  // Analyze outsourcing decisions
  const totalOps = currentSchedule.length;
  const outsourcedOps = currentSchedule.filter(op => 
    op.Assignment_Type === 'OUTSOURCE' || (!op.Machine_ID && op.Outsource_Cost > 0)
  );
  const inHouseOps = currentSchedule.filter(op => 
    op.Assignment_Type === 'IN_HOUSE' || op.Machine_ID
  );

  const outsourcedCount = outsourcedOps.length;
  const inHouseCount = inHouseOps.length;
  const outsourcingRate = totalOps > 0 ? (outsourcedCount / totalOps) * 100 : 0;

  // Cost analysis
  const totalOutsourceCost = outsourcedOps.reduce((sum, op) => sum + (op.Outsource_Cost || 0), 0);
  const totalInHouseCost = inHouseOps.reduce((sum, op) => {
    const duration = op.End_Time - op.Start_Time;
    return sum + (duration / 60 * 30); // Assuming $30/hr
  }, 0);
  const totalCost = totalOutsourceCost + totalInHouseCost;

  // Time analysis
  const avgOutsourceTime = outsourcedOps.length > 0 
    ? outsourcedOps.reduce((sum, op) => sum + (op.Outsource_Time_Min || 0), 0) / outsourcedOps.length
    : 0;
  const avgInHouseTime = inHouseOps.length > 0
    ? inHouseOps.reduce((sum, op) => sum + (op.End_Time - op.Start_Time), 0) / inHouseOps.length
    : 0;

  // Tardiness comparison
  const outsourcedLate = outsourcedOps.filter(op => op.Tardiness > 0).length;
  const inHouseLate = inHouseOps.filter(op => op.Tardiness > 0).length;
  const outsourcedOnTimeRate = outsourcedOps.length > 0 ? ((outsourcedOps.length - outsourcedLate) / outsourcedOps.length) * 100 : 100;
  const inHouseOnTimeRate = inHouseOps.length > 0 ? ((inHouseOps.length - inHouseLate) / inHouseOps.length) * 100 : 100;

  // Group outsourced operations by reason (we'll need to enhance this with backend data)
  const outsourcingReasons = {
    'Cost Effective': outsourcedOps.filter(op => {
      const inHouseCost = (op.End_Time - op.Start_Time) / 60 * 30;
      return (op.Outsource_Cost || 0) < inHouseCost * 0.9;
    }).length,
    'Deadline Constraint': outsourcedOps.filter(op => op.Tardiness === 0 && op.Due_Time < op.End_Time).length,
    'No Eligible Machines': outsourcedOps.filter(op => !op.Machine_ID).length,
    'Other': 0,
  };
  outsourcingReasons['Other'] = outsourcedCount - Object.values(outsourcingReasons).reduce((a, b) => a + b, 0);

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h1" gutterBottom>
          💰 Outsourcing Cost Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Comprehensive analysis of make-or-buy decisions for {currentHeuristic} heuristic
        </Typography>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>
                    Outsourced
                  </Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                    {outsourcedCount}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                    {outsourcingRate.toFixed(1)}% of total
                  </Typography>
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
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>
                    In-House
                  </Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                    {inHouseCount}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                    {(100 - outsourcingRate).toFixed(1)}% of total
                  </Typography>
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
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>
                    Outsource Cost
                  </Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                    ${totalOutsourceCost.toFixed(0)}
                  </Typography>
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
                  <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>
                    In-House Cost
                  </Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                    ${totalInHouseCost.toFixed(0)}
                  </Typography>
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

      {/* Performance Comparison */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ⚡ Performance Comparison
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Outsourced On-Time Rate</Typography>
                    <Typography variant="body2" fontWeight="bold">{outsourcedOnTimeRate.toFixed(1)}%</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={outsourcedOnTimeRate} 
                    sx={{ height: 10, borderRadius: 5 }}
                    color={outsourcedOnTimeRate >= 90 ? 'success' : 'warning'}
                  />
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">In-House On-Time Rate</Typography>
                    <Typography variant="body2" fontWeight="bold">{inHouseOnTimeRate.toFixed(1)}%</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={inHouseOnTimeRate} 
                    sx={{ height: 10, borderRadius: 5 }}
                    color={inHouseOnTimeRate >= 90 ? 'success' : 'warning'}
                  />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Avg Outsource Time</Typography>
                    <Typography variant="h6">{avgOutsourceTime.toFixed(0)} min</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Avg In-House Time</Typography>
                    <Typography variant="h6">{avgInHouseTime.toFixed(0)} min</Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Outsourcing Reasons
              </Typography>
              <Box sx={{ mt: 2 }}>
                {Object.entries(outsourcingReasons).map(([reason, count]) => (
                  <Box key={reason} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{reason}</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {count} ({outsourcedCount > 0 ? ((count / outsourcedCount) * 100).toFixed(0) : 0}%)
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={outsourcedCount > 0 ? (count / outsourcedCount) * 100 : 0} 
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Outsourced Operations Table */}
      {outsourcedOps.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📋 Outsourced Operations Details
            </Typography>
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
                    <TableCell align="right"><strong>Time (min)</strong></TableCell>
                    <TableCell><strong>Status</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {outsourcedOps.map((op, index) => {
                    const estInHouseCost = ((op.End_Time - op.Start_Time) / 60) * 30;
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
                            {savings > 0 ? (
                              <>
                                <TrendingDownIcon sx={{ fontSize: 16, color: 'success.main' }} />
                                <Typography variant="body2" color="success.main" fontWeight="bold">
                                  ${savings.toFixed(2)} ({savingsPercent.toFixed(0)}%)
                                </Typography>
                              </>
                            ) : (
                              <>
                                <TrendingUpIcon sx={{ fontSize: 16, color: 'error.main' }} />
                                <Typography variant="body2" color="error.main" fontWeight="bold">
                                  ${Math.abs(savings).toFixed(2)} ({Math.abs(savingsPercent).toFixed(0)}%)
                                </Typography>
                              </>
                            )}
                          </Box>
                        </TableCell>
                        <TableCell align="right">{op.Outsource_Time_Min || 'N/A'}</TableCell>
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

      {/* Strategic Insights */}
      <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <CardContent>
          <Typography variant="h6" sx={{ color: 'white', mb: 2 }}>
            💡 Strategic Insights & Recommendations
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Alert severity="info" sx={{ bgcolor: 'rgba(255,255,255,0.9)' }}>
                <Typography variant="subtitle2" fontWeight="bold">Outsourcing Rate Analysis</Typography>
                <Typography variant="body2">
                  {outsourcingRate > 50 
                    ? `High outsourcing rate (${outsourcingRate.toFixed(1)}%). Consider expanding in-house capacity or negotiating better vendor rates.`
                    : outsourcingRate > 20
                    ? `Moderate outsourcing (${outsourcingRate.toFixed(1)}%). Good balance between in-house and external resources.`
                    : `Low outsourcing rate (${outsourcingRate.toFixed(1)}%). In-house capacity is being well utilized.`
                  }
                </Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} md={6}>
              <Alert severity={totalOutsourceCost > totalInHouseCost ? 'warning' : 'success'} sx={{ bgcolor: 'rgba(255,255,255,0.9)' }}>
                <Typography variant="subtitle2" fontWeight="bold">Cost Efficiency</Typography>
                <Typography variant="body2">
                  {totalOutsourceCost > totalInHouseCost
                    ? `Outsourcing costs (${((totalOutsourceCost / totalCost) * 100).toFixed(1)}%) exceed in-house costs. Review vendor contracts and explore cost reduction opportunities.`
                    : `In-house production is cost-effective. Current outsourcing strategy is optimized for cost savings.`
                  }
                </Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} md={6}>
              <Alert severity={outsourcedOnTimeRate >= inHouseOnTimeRate ? 'success' : 'warning'} sx={{ bgcolor: 'rgba(255,255,255,0.9)' }}>
                <Typography variant="subtitle2" fontWeight="bold">Delivery Performance</Typography>
                <Typography variant="body2">
                  {outsourcedOnTimeRate >= inHouseOnTimeRate
                    ? `Outsourced operations have better on-time performance (${outsourcedOnTimeRate.toFixed(1)}% vs ${inHouseOnTimeRate.toFixed(1)}%). Vendors are reliable.`
                    : `In-house operations outperform outsourced (${inHouseOnTimeRate.toFixed(1)}% vs ${outsourcedOnTimeRate.toFixed(1)}%). Consider reviewing vendor SLAs.`
                  }
                </Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} md={6}>
              <Alert severity="info" sx={{ bgcolor: 'rgba(255,255,255,0.9)' }}>
                <Typography variant="subtitle2" fontWeight="bold">Capacity Planning</Typography>
                <Typography variant="body2">
                  {outsourcingReasons['No Eligible Machines'] > 0
                    ? `${outsourcingReasons['No Eligible Machines']} operations outsourced due to lack of machines. Consider capacity expansion for these operation types.`
                    : `All operations have eligible machines. Current capacity is sufficient for demand.`
                  }
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Container>
  );
}

export default OutsourcingAnalysis;
