import React, { useState } from 'react';
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
  TextField,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CalculateIcon from '@mui/icons-material/Calculate';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import api from '../services/api';

const CapexAnalysis = () => {
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [buyingMachine, setBuyingMachine] = useState(null);
  const [laborRate, setLaborRate] = useState(30);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setError(null);
    setSuccessMsg(null);
    
    try {
      const response = await api.post('/api/capex/analyze', null, {
        params: { hourly_labor_rate: laborRate }
      });
      
      if (response.data.status === 'success') {
        setAnalysis(response.data);
      } else {
        setError('Analysis failed: ' + (response.data.message || 'Unknown error'));
      }
    } catch (err) {
      setError('Failed to analyze CapEx opportunities: ' + (err.response?.data?.detail || err.message));
    } finally {
      setAnalyzing(false);
    }
  };

  const handleBuyMachine = async (machineId) => {
    setBuyingMachine(machineId);
    setError(null);
    setSuccessMsg(null);
    
    try {
      const response = await api.post('/api/capex/buy-machine', {
        machine_id: machineId,
        hourly_labor_rate: laborRate
      });
      
      if (response.data.status === 'success') {
        setSuccessMsg(`✅ ${response.data.message}`);
        // Re-run analysis to update recommendations
        setTimeout(() => {
          handleAnalyze();
        }, 1000);
      } else {
        setError('Purchase failed: ' + (response.data.message || 'Unknown error'));
      }
    } catch (err) {
      setError('Failed to purchase machine: ' + (err.response?.data?.detail || err.message));
    } finally {
      setBuyingMachine(null);
    }
  };

  const formatCurrency = (value) => {
    if (value == null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const getPaybackColor = (years) => {
    if (!years) return 'error';
    if (years < 1) return 'success';
    if (years < 3) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TrendingUpIcon fontSize="large" />
        Capital Expenditure Analysis
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Analyze outsourced operations to identify capital equipment purchase opportunities
        with ROI calculations and payback period estimates.
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <TextField
              label="Hourly Labor Rate ($)"
              type="number"
              value={laborRate}
              onChange={(e) => setLaborRate(parseFloat(e.target.value) || 30)}
              size="small"
              sx={{ width: 200 }}
            />
            
            <Button
              variant="contained"
              color="primary"
              onClick={handleAnalyze}
              disabled={analyzing}
              startIcon={analyzing ? <CircularProgress size={20} /> : <TrendingUpIcon />}
            >
              {analyzing ? 'Analyzing...' : 'Analyze CapEx Opportunities'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Messages */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {successMsg && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMsg(null)}>
          {successMsg}
        </Alert>
      )}

      {/* Analysis Results */}
      {analysis && (
        <>
          {/* Summary Card */}
          <Card sx={{ mb: 3, bgcolor: 'primary.50' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Summary
              </Typography>
              
              {analysis.biggest_offender ? (
                <Box>
                  <Typography variant="body1">
                    <strong>Biggest Offender:</strong>{' '}
                    <Chip 
                      label={analysis.biggest_offender} 
                      color="warning" 
                      size="small"
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    <strong>Outsourced Operations:</strong> {analysis.offender_count}
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    <strong>Total Vendor Cost:</strong> {formatCurrency(analysis.total_vendor_cost)}
                  </Typography>
                </Box>
              ) : (
                <Alert severity="info">
                  {analysis.message || 'No outsourced operations found.'}
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Recommendations Table */}
          {analysis.recommendations && analysis.recommendations.length > 0 && (
            <>
              {/* AI Explanation */}
              {analysis.ai_explanation && (
                <Card sx={{ mb: 3, bgcolor: 'info.50' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <SmartToyIcon color="primary" />
                      <Typography variant="h6">
                        AI Financial Analysis
                      </Typography>
                    </Box>
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                      {typeof analysis.ai_explanation === 'object' ? analysis.ai_explanation.text : analysis.ai_explanation}
                    </Typography>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Machine Purchase Recommendations
                  </Typography>
                  
                  <TableContainer component={Paper} sx={{ mt: 2 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow sx={{ bgcolor: 'grey.100' }}>
                          <TableCell><strong>Machine ID</strong></TableCell>
                          <TableCell><strong>Type</strong></TableCell>
                          <TableCell align="right"><strong>Purchase Price</strong></TableCell>
                          <TableCell align="right"><strong>In-House Cost</strong></TableCell>
                          <TableCell align="right"><strong>Vendor Cost</strong></TableCell>
                          <TableCell align="right"><strong>Annual Savings</strong></TableCell>
                          <TableCell align="right"><strong>Payback (Years)</strong></TableCell>
                          <TableCell align="right"><strong>Jobs</strong></TableCell>
                          {/* Action column (Buy) removed temporarily */}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {analysis.recommendations.map((rec) => (
                          <React.Fragment key={rec.machine_id}>
                            <TableRow 
                              sx={{ 
                                '&:hover': { bgcolor: 'action.hover' },
                                bgcolor: rec.savings > 0 ? 'success.50' : 'error.50'
                              }}
                            >
                              <TableCell>{rec.machine_id}</TableCell>
                              <TableCell>{rec.machine_type}</TableCell>
                              <TableCell align="right">{formatCurrency(rec.purchase_price)}</TableCell>
                              <TableCell align="right">
                                {formatCurrency(rec.total_inhouse_cost)}
                                <Typography variant="caption" display="block" color="text.secondary">
                                  Labor: {formatCurrency(rec.labor_cost)} + Energy: {formatCurrency(rec.energy_cost)}
                                </Typography>
                              </TableCell>
                              <TableCell align="right">{formatCurrency(rec.vendor_cost)}</TableCell>
                              <TableCell align="right">
                                <Typography 
                                  variant="body2" 
                                  sx={{ 
                                    color: rec.savings > 0 ? 'success.main' : 'error.main',
                                    fontWeight: 'bold'
                                  }}
                                >
                                  {formatCurrency(rec.savings)}
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                {rec.payback_years ? (
                                  <Chip 
                                    label={`${rec.payback_years} yrs`}
                                    color={getPaybackColor(rec.payback_years)}
                                    size="small"
                                  />
                                ) : (
                                  <Chip label="No ROI" color="error" size="small" />
                                )}
                              </TableCell>
                              <TableCell align="right">
                                {rec.jobs_count}
                                <Typography variant="caption" display="block" color="text.secondary">
                                  {rec.total_proc_hours.toFixed(1)} hrs
                                </Typography>
                              </TableCell>
                              {/* Buy button removed temporarily */}
                            </TableRow>
                            {/* Expandable Calculation Details */}
                            {rec.calculation_details && (
                              <TableRow>
                                <TableCell colSpan={9} sx={{ py: 0, borderBottom: 'none' }}>
                                  <Accordion elevation={0}>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <CalculateIcon fontSize="small" color="action" />
                                        <Typography variant="body2" color="text.secondary">
                                          View Calculation Details
                                        </Typography>
                                      </Box>
                                    </AccordionSummary>
                                    <AccordionDetails>
                                      <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                                        <Typography variant="subtitle2" gutterBottom>
                                          <strong>Calculation Breakdown:</strong>
                                        </Typography>
                                        
                                        <Box sx={{ mt: 2 }}>
                                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                                            <strong>Labor Cost:</strong> {rec.calculation_details.formula.labor}
                                          </Typography>
                                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                                            <strong>Energy Cost:</strong> {rec.calculation_details.formula.energy}
                                          </Typography>
                                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                                            <strong>Total In-House:</strong> {rec.calculation_details.formula.total_inhouse}
                                          </Typography>
                                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                                            <strong>Savings:</strong> {rec.calculation_details.formula.savings}
                                          </Typography>
                                          <Typography variant="body2" sx={{ mb: 0.5, color: rec.savings > 0 ? 'success.main' : 'error.main', fontWeight: 'bold' }}>
                                            <strong>Payback:</strong> {rec.calculation_details.formula.payback}
                                          </Typography>
                                        </Box>

                                        <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                                          <Typography variant="caption" color="text.secondary">
                                            <strong>Assumptions:</strong> Hourly rate ${rec.calculation_details.hourly_labor_rate}/hr, 
                                            Energy = 10% of labor, Speed factor = {rec.calculation_details.speed_factor}
                                          </Typography>
                                        </Box>
                                      </Box>
                                    </AccordionDetails>
                                  </Accordion>
                                </TableCell>
                              </TableRow>
                            )}
                          </React.Fragment>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>

                  <Alert severity="info" sx={{ mt: 2 }}>
                    <strong>How to use:</strong> Click "Buy" to clone the selected machine and add it to your fleet.
                    This will create a new machine instance with a unique ID (e.g., M1_NEW1) and permanently
                    update your machine_data.csv file. The new machine will be available for scheduling immediately.
                  </Alert>
                </CardContent>
              </Card>
            </>
          )}
        </>
      )}

      {/* Empty State */}
      {!analysis && !analyzing && (
        <Card sx={{ textAlign: 'center', py: 6 }}>
          <CardContent>
            <TrendingUpIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Analysis Yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Click "Analyze CapEx Opportunities" to identify machine purchase recommendations
              based on your current outsourcing patterns.
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default CapexAnalysis;
