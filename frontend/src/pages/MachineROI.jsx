import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Grid,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  AttachMoney as MoneyIcon,
  Speed as SpeedIcon,
  Build as BuildIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Assessment as AssessmentIcon,
  Star as StarIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import api from '../services/api';

const MachineROI = () => {
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadAnalysis();
  }, []);

  const loadAnalysis = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.get('/api/analysis/machine-roi');
      
      if (response.data.status === 'success') {
        setAnalysis(response.data);
      } else {
        setError('Analysis failed: ' + (response.data.message || 'Unknown error'));
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message;
      setError('Failed to load machine ROI analysis: ' + errorMsg);
      enqueueSnackbar(errorMsg, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    if (value == null || isNaN(value)) return '$0';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercent = (value) => {
    if (value == null || isNaN(value)) return '0%';
    return `${value.toFixed(1)}%`;
  };

  const getUtilizationColor = (pct) => {
    if (pct >= 80) return 'success';
    if (pct >= 50) return 'warning';
    return 'error';
  };

  const getROIColor = (roi) => {
    if (roi >= 50) return 'success';
    if (roi >= 20) return 'warning';
    if (roi > 0) return 'info';
    return 'error';
  };

  const getPriorityColor = (priority) => {
    if (priority === 'HIGH') return 'error';
    if (priority === 'MEDIUM') return 'warning';
    return 'info';
  };

  const getRecommendationIcon = (recommendation) => {
    switch (recommendation) {
      case 'EXPAND':
        return <StarIcon />;
      case 'OPTIMIZE':
        return <BuildIcon />;
      case 'REVIEW':
        return <WarningIcon />;
      case 'CRITICAL_REVIEW':
        return <ErrorIcon />;
      default:
        return <AssessmentIcon />;
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AssessmentIcon fontSize="large" />
            Machine ROI & Investment Analysis
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Comprehensive analysis of machine utilization, profitability, and investment recommendations
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={loadAnalysis}
          disabled={loading}
        >
          Refresh Analysis
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {analysis && (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={3}>
              <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                      <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>
                        Total Profit
                      </Typography>
                      <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                        {formatCurrency(analysis.summary.total_profit)}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                        {analysis.analysis_period_days.toFixed(1)} day period
                      </Typography>
                    </Box>
                    <MoneyIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
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
                        Avg Utilization
                      </Typography>
                      <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                        {formatPercent(analysis.summary.avg_utilization_pct)}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                        {analysis.summary.active_machines}/{analysis.summary.total_machines} machines active
                      </Typography>
                    </Box>
                    <SpeedIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
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
                        Total Revenue
                      </Typography>
                      <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                        {formatCurrency(analysis.summary.total_revenue)}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                        Generated value
                      </Typography>
                    </Box>
                    <TrendingUpIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
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
                        Best ROI
                      </Typography>
                      <Typography variant="h3" sx={{ color: 'white', fontWeight: 'bold' }}>
                        {formatPercent(analysis.summary.highest_roi_value)}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                        {analysis.summary.highest_roi_machine || 'N/A'}
                      </Typography>
                    </Box>
                    <StarIcon sx={{ fontSize: 60, color: 'white', opacity: 0.3 }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Investment Recommendations */}
          {analysis.recommendations && analysis.recommendations.length > 0 && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <BuildIcon />
                  Investment Recommendations
                </Typography>
                
                <Box sx={{ mt: 2 }}>
                  {analysis.recommendations.map((rec, index) => (
                    <Accordion key={index}>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                          {getRecommendationIcon(rec.recommendation)}
                          <Typography sx={{ fontWeight: 'bold' }}>{rec.machine_id}</Typography>
                          <Chip 
                            label={rec.recommendation.replace('_', ' ')} 
                            color={getPriorityColor(rec.priority)}
                            size="small"
                          />
                          <Chip 
                            label={rec.priority} 
                            variant="outlined"
                            size="small"
                          />
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="body2" paragraph>
                          {rec.reason}
                        </Typography>
                        {rec.estimated_additional_revenue && (
                          <Typography variant="body2" color="success.main">
                            <strong>Potential Additional Revenue:</strong> {formatCurrency(rec.estimated_additional_revenue)}/year
                          </Typography>
                        )}
                        {rec.potential_savings && (
                          <Typography variant="body2" color="warning.main">
                            <strong>Potential Savings:</strong> {formatCurrency(rec.potential_savings)}
                          </Typography>
                        )}
                        {rec.annual_loss && (
                          <Typography variant="body2" color="error.main">
                            <strong>Annual Loss:</strong> {formatCurrency(rec.annual_loss)}
                          </Typography>
                        )}
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Detailed Machine Metrics Table */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detailed Machine Analysis
              </Typography>
              
              <TableContainer component={Paper} sx={{ mt: 2, maxHeight: 600 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'grey.100' }}>
                      <TableCell><strong>Machine</strong></TableCell>
                      <TableCell><strong>Type</strong></TableCell>
                      <TableCell align="right"><strong>Utilization</strong></TableCell>
                      <TableCell align="right"><strong>Jobs</strong></TableCell>
                      <TableCell align="right"><strong>Active Hours</strong></TableCell>
                      <TableCell align="right"><strong>Revenue</strong></TableCell>
                      <TableCell align="right"><strong>Operating Cost</strong></TableCell>
                      <TableCell align="right"><strong>Profit</strong></TableCell>
                      <TableCell align="right"><strong>ROI %</strong></TableCell>
                      <TableCell align="right"><strong>Payback</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {analysis.machines.map((machine) => (
                      <React.Fragment key={machine.machine_id}>
                        <TableRow 
                          sx={{ 
                            '&:hover': { bgcolor: 'action.hover' },
                            bgcolor: machine.roi_pct > 50 ? 'success.50' : machine.roi_pct < 0 ? 'error.50' : 'inherit'
                          }}
                        >
                          <TableCell>
                            <Typography variant="body2" fontWeight="bold">
                              {machine.machine_id}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="caption">
                              {machine.machine_type}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Box sx={{ minWidth: 120 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body2">
                                  {formatPercent(machine.utilization_pct)}
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={Math.min(machine.utilization_pct, 100)}
                                color={getUtilizationColor(machine.utilization_pct)}
                                sx={{ mt: 0.5 }}
                              />
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            {machine.jobs_count}
                            <Typography variant="caption" display="block" color="text.secondary">
                              {machine.operations_count} ops
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            {machine.total_active_hours.toFixed(1)}
                            <Typography variant="caption" display="block" color="text.secondary">
                              Idle: {machine.idle_hours.toFixed(1)}h
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" color="success.main" fontWeight="bold">
                              {formatCurrency(machine.revenue)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" color="error.main">
                              {formatCurrency(machine.total_operating_cost)}
                            </Typography>
                            <Typography variant="caption" display="block" color="text.secondary">
                              Labor: {formatCurrency(machine.labor_cost)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography 
                              variant="body2" 
                              fontWeight="bold"
                              color={machine.profit > 0 ? 'success.main' : 'error.main'}
                            >
                              {formatCurrency(machine.profit)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={formatPercent(machine.roi_pct)}
                              color={getROIColor(machine.roi_pct)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell align="right">
                            {machine.payback_years ? (
                              <Typography variant="body2">
                                {machine.payback_years.toFixed(1)} yrs
                              </Typography>
                            ) : (
                              <Typography variant="body2" color="text.secondary">
                                N/A
                              </Typography>
                            )}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell colSpan={10} sx={{ py: 0, borderBottom: 'none' }}>
                            <Accordion elevation={0}>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Typography variant="caption" color="text.secondary">
                                  View Detailed Metrics & Breakdown
                                </Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                <Grid container spacing={2}>
                                  <Grid item xs={12} md={6}>
                                    <Typography variant="subtitle2" gutterBottom>
                                      Operational Metrics
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Speed Factor:</strong> {machine.speed_factor}x
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Processing Hours:</strong> {machine.total_proc_hours.toFixed(1)}h
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Setup Hours:</strong> {machine.total_setup_hours.toFixed(1)}h
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Maintenance Hours:</strong> {machine.maintenance_hours.toFixed(1)}h
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Machine Span:</strong> {machine.machine_span_hours.toFixed(1)}h
                                    </Typography>
                                  </Grid>
                                  <Grid item xs={12} md={6}>
                                    <Typography variant="subtitle2" gutterBottom>
                                      Financial Breakdown
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Purchase Price:</strong> {formatCurrency(machine.purchase_price)}
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Energy Cost:</strong> {formatCurrency(machine.energy_cost)}
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Maintenance Cost:</strong> {formatCurrency(machine.maintenance_cost)}
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Annual Profit Projection:</strong> {formatCurrency(machine.annual_profit)}
                                    </Typography>
                                  </Grid>
                                  <Grid item xs={12}>
                                    <Typography variant="subtitle2" gutterBottom>
                                      Performance
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Throughput:</strong> {machine.throughput_ops_per_day.toFixed(2)} ops/day
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Avg Cycle Time:</strong> {machine.avg_cycle_time_hours.toFixed(2)} hours/operation
                                    </Typography>
                                    <Typography variant="body2">
                                      <strong>Operation Types:</strong> {machine.op_types_handled.join(', ') || 'None'}
                                    </Typography>
                                  </Grid>
                                </Grid>
                              </AccordionDetails>
                            </Accordion>
                          </TableCell>
                        </TableRow>
                      </React.Fragment>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Divider sx={{ my: 2 }} />

              <Typography variant="caption" color="text.secondary">
                <strong>Analysis Parameters:</strong> Hourly labor rate: ${analysis.parameters.hourly_labor_rate}/hr, 
                Energy cost: {analysis.parameters.energy_cost_rate_pct}% of labor, 
                Analysis period: {analysis.analysis_period_days.toFixed(1)} days,
                Heuristic: {analysis.heuristic}
              </Typography>
            </CardContent>
          </Card>
        </>
      )}

      {/* Empty State */}
      {!analysis && !loading && !error && (
        <Card sx={{ textAlign: 'center', py: 6 }}>
          <CardContent>
            <AssessmentIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Analysis Data
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Please load data and compute a heuristic schedule first.
            </Typography>
            <Button variant="contained" onClick={loadAnalysis}>
              Load Analysis
            </Button>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default MachineROI;
