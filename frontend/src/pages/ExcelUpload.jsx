// frontend/src/pages/ExcelUpload.jsx
import React, { useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Button,
  Box,
  Stepper,
  Step,
  StepLabel,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Card,
  CardContent,
  Grid,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  AutoFixHigh as AutoMapIcon,
  CheckCircle as ConfirmIcon,
  Psychology as AIIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import axios from 'axios';
import Plot from 'react-plotly.js';

const steps = ['Upload Excel', 'Map Columns', 'Confirm & Load'];

const API_BASE = 'http://localhost:8001';

function ExcelUpload() {
  const { enqueueSnackbar } = useSnackbar();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  
  // Step 1: Upload
  const [file, setFile] = useState(null);
  const [sheets, setSheets] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState('');
  
  // Step 2: Column Mapping
  const [columns, setColumns] = useState([]);
  const [mappings, setMappings] = useState({});
  const [availableFields, setAvailableFields] = useState([]);
  
  // Step 3: Results
  const [transformResult, setTransformResult] = useState(null);
  const [scheduleResult, setScheduleResult] = useState(null);
  const [selectedHeuristic, setSelectedHeuristic] = useState(null);
  const [aiInsightsOpen, setAiInsightsOpen] = useState(false);
  const [aiInsights, setAiInsights] = useState('');

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      uploadFile(selectedFile);
    }
  };

  const uploadFile = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE}/api/excel/upload`, formData);
      
      setSheets(response.data.sheet_names);
      setSelectedSheet(response.data.sheet_names[0]);
      
      enqueueSnackbar('File uploaded successfully!', { variant: 'success' });
      
      // Auto-proceed to mapping
      autoMapColumns(file, response.data.sheet_names[0]);
      
    } catch (error) {
      enqueueSnackbar(`Upload failed: ${error.response?.data?.detail || error.message}`, { 
        variant: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const autoMapColumns = async (file, sheetName) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (sheetName) formData.append('sheet_name', sheetName);
      formData.append('use_llm', 'true');
      
      const response = await axios.post(`${API_BASE}/api/excel/auto-map`, formData);
      
      console.log('Auto-map response:', response.data);
      
      const mappingData = response.data.mappings;
      console.log('Mapping data:', mappingData);
      
      setColumns(mappingData);
      
      // Set initial mappings
      const initialMappings = {};
      mappingData.forEach(m => {
        initialMappings[m.excel_column] = m.canonical_field;
      });
      console.log('Initial mappings:', initialMappings);
      setMappings(initialMappings);
      
      // Get available fields
      if (mappingData.length > 0) {
        const fields = mappingData[0].available_fields || [];
        console.log('Available fields:', fields);
        setAvailableFields(fields);
      }
      
      setActiveStep(1);
      enqueueSnackbar('Columns mapped automatically!', { variant: 'success' });
      
    } catch (error) {
      enqueueSnackbar(`Auto-mapping failed: ${error.response?.data?.detail || error.message}`, { 
        variant: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleMappingChange = (excelColumn, newField) => {
    setMappings(prev => ({
      ...prev,
      [excelColumn]: newField
    }));
  };

  const confirmAndTransform = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (selectedSheet) formData.append('sheet_name', selectedSheet);
      formData.append('mappings', JSON.stringify(mappings));
      formData.append('save_as_template', 'false');
      
      const response = await axios.post(`${API_BASE}/api/excel/transform`, formData);
      
      setTransformResult(response.data);
      setActiveStep(2);
      
      enqueueSnackbar(
        `Successfully transformed ${response.data.job_count} jobs! Ready to schedule.`, 
        { variant: 'success' }
      );
      
    } catch (error) {
      console.error('Transform error:', error.response?.data);
      
      let errorMessage = 'Transform failed';
      
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        if (typeof detail === 'object') {
          errorMessage = detail.message || 'Data transformation failed';
          if (detail.errors && detail.errors.length > 0) {
            errorMessage += `: ${detail.errors.slice(0, 3).join(', ')}`;
            if (detail.errors.length > 3) {
              errorMessage += ` (and ${detail.errors.length - 3} more)`;
            }
          }
        } else {
          errorMessage = detail;
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      enqueueSnackbar(errorMessage, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleLoadAndSchedule = async (heuristic) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (selectedSheet) formData.append('sheet_name', selectedSheet);
      formData.append('mappings', JSON.stringify(mappings));
      formData.append('heuristic', heuristic);
      
      const response = await axios.post(`${API_BASE}/api/excel/load-and-schedule`, formData);
      
      setSelectedHeuristic(heuristic);
      setScheduleResult(response.data);
      
      enqueueSnackbar(
        `Scheduled ${response.data.job_count} jobs using ${heuristic}!`, 
        { variant: 'success' }
      );
      
      // Auto-fetch AI insights after scheduling
      setTimeout(async () => {
        try {
          const aiResponse = await axios.post(`${API_BASE}/api/ai/insights`, {
            prompt: `Analyze the ${heuristic} scheduling results:\n- Evaluate key performance metrics (makespan, tardiness, total cost, utilization)\n- Identify potential bottlenecks or inefficiencies\n- Provide specific recommendations for improvement\n\nMetrics: ${JSON.stringify(response.data.metrics)}`,
            context_data: response.data.metrics
          });
          
          setAiInsights(cleanAIInsights(aiResponse.data.insights));
        } catch (error) {
          console.error('Failed to fetch AI insights:', error);
        }
      }, 500);
      
    } catch (error) {
      console.error('Scheduling error:', error);
      
      let errorMessage = 'Scheduling failed';
      if (error.response?.data?.detail) {
        errorMessage = typeof error.response.data.detail === 'string' 
          ? error.response.data.detail 
          : JSON.stringify(error.response.data.detail);
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      enqueueSnackbar(errorMessage, { 
        variant: 'error',
        autoHideDuration: 10000
      });
    } finally {
      setLoading(false);
    }
  };

  const handleComputeAllHeuristics = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/api/schedule/compute-all`);
      
      enqueueSnackbar('All heuristics computed successfully!', { variant: 'success' });
      
      // Fetch comparison data
      const comparisonResponse = await axios.get(`${API_BASE}/api/metrics/comparison`);
      setScheduleResult({
        ...scheduleResult,
        allMetrics: comparisonResponse.data.results
      });
      
    } catch (error) {
      enqueueSnackbar(`Failed to compute: ${error.response?.data?.detail || error.message}`, { 
        variant: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleGetAIInsights = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/api/metrics/comparison`);
      const metricsData = response.data.results;
      
      const aiResponse = await axios.post(`${API_BASE}/api/ai/insights`, {
        prompt: `Evaluate the performance of these scheduling heuristics:\n- Identify the best performing heuristic and explain why\n- Highlight specific metric differences (makespan, tardiness, cost, utilization)\n- Note any trade-offs between different objectives\n- Recommend which heuristic to use for this dataset\n\nResults:\n${JSON.stringify(metricsData, null, 2)}`,
        context_data: metricsData
      });
      
      setAiInsights(cleanAIInsights(aiResponse.data.insights));
      setAiInsightsOpen(true);
      
    } catch (error) {
      enqueueSnackbar(`AI insights failed: ${error.response?.data?.detail || error.message}`, { 
        variant: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.85) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  const cleanAIInsights = (text) => {
    if (!text) return '';
    
    let cleaned = text;
    
    // Remove markdown tables (lines with | characters)
    cleaned = cleaned.split('\n')
      .filter(line => !line.trim().startsWith('|') && !line.includes('---|'))
      .join('\n');
    
    // Remove ** bold markers
    cleaned = cleaned.replace(/\*\*/g, '');
    
    // Remove extra blank lines (more than 1 consecutive)
    cleaned = cleaned.replace(/\n\s*\n\s*\n/g, '\n\n');
    
    // Remove leading/trailing whitespace
    cleaned = cleaned.trim();
    
    return cleaned;
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h1" gutterBottom>
        Excel Data Import
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload any Excel file and our AI will automatically understand your data format
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {/* Step 1: Upload */}
      {activeStep === 0 && (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <UploadIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
          
          <Typography variant="h6" gutterBottom>
            Select Excel File
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Supports .xlsx and .xls formats
          </Typography>
          
          <Button
            variant="contained"
            component="label"
            startIcon={<UploadIcon />}
            size="large"
          >
            Choose File
            <input
              type="file"
              hidden
              accept=".xlsx,.xls"
              onChange={handleFileSelect}
            />
          </Button>
          
          {file && (
            <Alert severity="info" sx={{ mt: 2 }}>
              Selected: {file.name}
            </Alert>
          )}
          
          {loading && <CircularProgress sx={{ mt: 2 }} />}
        </Paper>
      )}

      {/* Step 2: Column Mapping */}
      {activeStep === 1 && (
        <Paper sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h6">
              Review Column Mappings
            </Typography>
            <Button
              variant="contained"
              startIcon={<ConfirmIcon />}
              onClick={confirmAndTransform}
              disabled={loading}
            >
              Confirm & Transform
            </Button>
          </Box>
          
          <Alert severity="info" sx={{ mb: 2 }}>
            Our AI has automatically mapped your columns. Review and adjust if needed.
          </Alert>
          
          {columns.length === 0 ? (
            <Alert severity="warning">
              No column mappings found. Please try uploading the file again.
            </Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Excel Column</TableCell>
                    <TableCell>Detected As</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Source</TableCell>
                    <TableCell>Correct Mapping</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {columns.map((col) => (
                    <TableRow key={col.excel_column}>
                      <TableCell>
                        <strong>{col.excel_column}</strong>
                      </TableCell>
                      <TableCell>{col.canonical_field}</TableCell>
                      <TableCell>
                        <Chip
                          label={`${Math.round(col.confidence * 100)}%`}
                          color={getConfidenceColor(col.confidence)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip label={col.source} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>
                        <FormControl fullWidth size="small">
                          <Select
                            value={mappings[col.excel_column] || col.canonical_field}
                            onChange={(e) => handleMappingChange(col.excel_column, e.target.value)}
                          >
                            {(availableFields.length > 0 ? availableFields : col.available_fields || []).map(field => (
                              <MenuItem key={field} value={field}>
                                {field}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <CircularProgress />
            </Box>
          )}
        </Paper>
      )}

      {/* Step 3: Results */}
      {activeStep === 2 && transformResult && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            ✅ Transformation Complete
          </Typography>
          
          <Alert severity="success" sx={{ mb: 2 }}>
            Successfully transformed {transformResult.job_count} jobs!
          </Alert>
          
          {transformResult.warnings && transformResult.warnings.length > 0 && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="subtitle2">Warnings:</Typography>
              <ul>
                {transformResult.warnings.slice(0, 5).map((warn, i) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </Alert>
          )}
          
          {transformResult.errors && transformResult.errors.length > 0 && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="subtitle2">Errors:</Typography>
              <ul>
                {transformResult.errors.slice(0, 5).map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </Alert>
          )}
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 3 }}>
            Preview of loaded jobs (first 5):
          </Typography>
          
          <TableContainer sx={{ mt: 2 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Job ID</TableCell>
                  <TableCell>Processing Time</TableCell>
                  <TableCell>Due Date</TableCell>
                  <TableCell>Machine</TableCell>
                  <TableCell>Priority</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {transformResult.jobs.slice(0, 5).map((job, i) => (
                  <TableRow key={i}>
                    <TableCell>{job.job_id}</TableCell>
                    <TableCell>{job.processing_time}hrs</TableCell>
                    <TableCell>{job.due_date || 'N/A'}</TableCell>
                    <TableCell>{job.machine || 'N/A'}</TableCell>
                    <TableCell>{job.priority || 'N/A'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Schedule Jobs
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Choose a scheduling algorithm to compute optimal job sequences:
            </Typography>
            
            {loading && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CircularProgress size={20} />
                  <Typography variant="body2">
                    Processing your data and running scheduling algorithm... This may take a few moments.
                  </Typography>
                </Box>
              </Alert>
            )}
            
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <ConfirmIcon />}
                onClick={() => handleLoadAndSchedule('SPT')}
                disabled={loading}
                sx={{ 
                  flex: 1,
                  minWidth: 200,
                  backgroundColor: '#10b981',
                  '&:hover': { backgroundColor: '#059669' }
                }}
              >
                SPT - Shortest Processing Time
              </Button>
              
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <ConfirmIcon />}
                onClick={() => handleLoadAndSchedule('EDD')}
                disabled={loading}
                sx={{ 
                  flex: 1,
                  minWidth: 200,
                  backgroundColor: '#3b82f6',
                  '&:hover': { backgroundColor: '#2563eb' }
                }}
              >
                EDD - Earliest Due Date
              </Button>
            </Box>
            
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <ConfirmIcon />}
                onClick={() => handleLoadAndSchedule('CR')}
                disabled={loading}
                sx={{ 
                  flex: 1,
                  minWidth: 200,
                  backgroundColor: '#f59e0b',
                  '&:hover': { backgroundColor: '#d97706' }
                }}
              >
                CR - Critical Ratio
              </Button>
              
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <ConfirmIcon />}
                onClick={() => handleLoadAndSchedule('PRIORITY')}
                disabled={loading}
                sx={{ 
                  flex: 1,
                  minWidth: 200,
                  backgroundColor: '#8b5cf6',
                  '&:hover': { backgroundColor: '#7c3aed' }
                }}
              >
                PRIORITY - Priority Based
              </Button>
            </Box>
            
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, fontStyle: 'italic' }}>
              Tip: Each algorithm optimizes for different criteria. SPT minimizes average completion time, 
              EDD reduces tardiness, CR balances urgency and time, PRIORITY respects job priorities.
            </Typography>
            
            <Button
              variant="outlined"
              onClick={() => {
                setActiveStep(0);
                setFile(null);
                setColumns([]);
                setMappings({});
                setTransformResult(null);
                setScheduleResult(null);
                setSelectedHeuristic(null);
              }}
              sx={{ mt: 2, alignSelf: 'flex-start' }}
            >
              Upload Another File
            </Button>
          </Box>
        </Paper>
      )}

      {/* Step 4: Schedule Results with Gantt Chart */}
      {scheduleResult && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h5" gutterBottom>
            Scheduling Results - {selectedHeuristic}
          </Typography>

          <Alert severity="success" sx={{ mb: 3 }}>
            Successfully scheduled {scheduleResult.job_count} jobs using {selectedHeuristic} algorithm!
          </Alert>

          {/* Detailed KPI Table for Selected Heuristic */}
          <Card sx={{ mb: 3, bgcolor: '#f8f9fa' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                📊 Performance Metrics - {selectedHeuristic}
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Makespan
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {scheduleResult.metrics?.Makespan_Days?.toFixed(2) || scheduleResult.metrics?.makespan?.toFixed(0) || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {scheduleResult.metrics?.Makespan_Days ? 'days' : 'min'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Tardiness
                    </Typography>
                    <Typography variant="h6" color="error">
                      {scheduleResult.metrics?.Total_Tardiness_Days?.toFixed(2) || scheduleResult.metrics?.total_tardiness?.toFixed(0) || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {scheduleResult.metrics?.Total_Tardiness_Days ? 'days' : 'min'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Utilization
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      {scheduleResult.metrics?.['Machine_Utilization_%']?.toFixed(1) || scheduleResult.metrics?.utilization?.toFixed(1) || 'N/A'}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      machines
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      On-Time %
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      {scheduleResult.metrics?.['On_Time_%']?.toFixed(1) || scheduleResult.metrics?.on_time_pct?.toFixed(1) || 'N/A'}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      delivery
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Late Operations
                    </Typography>
                    <Typography variant="h6" color="warning.main">
                      {scheduleResult.metrics?.Late_Operations || scheduleResult.metrics?.late_ops || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      of {scheduleResult.metrics?.Total_Operations || scheduleResult.job_count}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Total Cost
                    </Typography>
                    <Typography variant="h6" color="primary">
                      ${scheduleResult.metrics?.['Total_Cost_$']?.toFixed(0) || scheduleResult.metrics?.total_cost?.toFixed(0) || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      estimated
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Avg Tardiness
                    </Typography>
                    <Typography variant="h6" color="warning.main">
                      {scheduleResult.metrics?.Avg_Tardiness_Min?.toFixed(1) || scheduleResult.metrics?.avg_tardiness?.toFixed(1) || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      min/op
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'white', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Algorithm
                    </Typography>
                    <Typography variant="h6" color="secondary">
                      {selectedHeuristic}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      heuristic
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Old KPI Cards - Remove or Keep for backward compatibility */}
          <Grid container spacing={2} sx={{ mb: 3, display: 'none' }}>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#e3f2fd' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Total Makespan
                  </Typography>
                  <Typography variant="h5">
                    {scheduleResult.metrics?.makespan?.toFixed(0) || 'N/A'} min
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#f3e5f5' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Avg Completion Time
                  </Typography>
                  <Typography variant="h5">
                    {scheduleResult.metrics?.avg_completion?.toFixed(0) || 'N/A'} min
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#fff3e0' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Total Tardiness
                  </Typography>
                  <Typography variant="h5">
                    {scheduleResult.metrics?.total_tardiness?.toFixed(0) || 'N/A'} min
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card sx={{ bgcolor: '#e8f5e9' }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    Machine Utilization
                  </Typography>
                  <Typography variant="h5">
                    {scheduleResult.metrics?.utilization?.toFixed(1) || 'N/A'}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Gantt Chart */}
          {scheduleResult.schedule && scheduleResult.schedule.length > 0 && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Gantt Chart Visualization
                </Typography>
                <Plot
                  data={scheduleResult.schedule.map((item) => {
                    // Generate unique color for each job (same logic as main Gantt)
                    const getJobColor = (jobId) => {
                      const colors = [
                        '#1976d2', '#d32f2f', '#388e3c', '#f57c00', '#7b1fa2',
                        '#0097a7', '#c2185b', '#5d4037', '#455a64', '#e64a19',
                        '#00796b', '#303f9f', '#c62828', '#6a1b9a', '#0277bd'
                      ];
                      let hash = 0;
                      for (let i = 0; i < jobId.length; i++) {
                        hash = jobId.charCodeAt(i) + ((hash << 5) - hash);
                      }
                      return colors[Math.abs(hash) % colors.length];
                    };
                    
                    return {
                      x: [item.Start_Time, item.End_Time],
                      y: [item.Machine_ID, item.Machine_ID],
                      type: 'scatter',
                      mode: 'lines',
                      line: { width: 20, color: getJobColor(item.Job_ID) },
                      name: `${item.Job_ID} - ${item.Operation_ID}`,
                      text: `${item.Job_ID}`,
                      hovertemplate:
                        `<b>Machine:</b> ${item.Machine_ID}<br>` +
                        `<b>Job:</b> ${item.Job_ID}<br>` +
                        `<b>Operation:</b> ${item.Operation_ID}<br>` +
                        `<b>Start:</b> ${item.Start_Time} min<br>` +
                        `<b>End:</b> ${item.End_Time} min<br>` +
                        `<b>Duration:</b> ${item.End_Time - item.Start_Time} min` +
                        (item.Priority ? `<br><b>Priority:</b> ${item.Priority}` : '') +
                        `<extra></extra>`,
                      hoverlabel: {
                        bgcolor: 'white',
                        font: { size: 12, color: 'black' }
                      }
                    };
                  })}
                  layout={{
                    title: `${selectedHeuristic} Schedule - Gantt Chart`,
                    xaxis: {
                      title: 'Time (minutes)',
                      showgrid: true,
                      zeroline: false,
                    },
                    yaxis: {
                      title: 'Machine',
                      autorange: 'reversed',
                    },
                    height: 600,
                    showlegend: false,
                    hovermode: 'closest',
                  }}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
                  }}
                  style={{ width: '100%' }}
                />
                <Alert severity="info" sx={{ mt: 2 }}>
                  <strong>Legend:</strong> Hover over bars to see operation details. Each job has a unique color for easy tracking across machines.
                </Alert>
              </CardContent>
            </Card>
          )}

          {/* AI Insights Panel */}
          {aiInsights && (
            <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <AIIcon sx={{ color: 'white', fontSize: 28 }} />
                  <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
                    ✨ AI Insights & Recommendations
                  </Typography>
                </Box>
                <Box sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.95)', 
                  p: 2, 
                  borderRadius: 2,
                  maxHeight: '400px',
                  overflowY: 'auto'
                }}>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.6 }}>
                    {aiInsights}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mt: 3 }}>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
              onClick={handleComputeAllHeuristics}
              disabled={loading}
              sx={{ backgroundColor: '#6366f1', '&:hover': { backgroundColor: '#4f46e5' } }}
            >
              Compute All Heuristics
            </Button>

            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <AIIcon />}
              onClick={handleGetAIInsights}
              disabled={loading}
              sx={{ backgroundColor: '#ec4899', '&:hover': { backgroundColor: '#db2777' } }}
            >
              AI Insights & Comparison
            </Button>

            <Button
              variant="outlined"
              onClick={() => {
                setActiveStep(0);
                setFile(null);
                setColumns([]);
                setMappings({});
                setTransformResult(null);
                setScheduleResult(null);
                setSelectedHeuristic(null);
              }}
            >
              Upload Another File
            </Button>
          </Box>

          {/* All Heuristics Comparison Table */}
          {scheduleResult.allMetrics && (
            <Box sx={{ mt: 3 }}>
              <Divider sx={{ my: 3 }} />
              <Typography variant="h6" gutterBottom>
                Heuristics Comparison
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Heuristic</strong></TableCell>
                      <TableCell align="right"><strong>Makespan (min)</strong></TableCell>
                      <TableCell align="right"><strong>Avg Completion (min)</strong></TableCell>
                      <TableCell align="right"><strong>Total Tardiness (min)</strong></TableCell>
                      <TableCell align="right"><strong>Utilization (%)</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(scheduleResult.allMetrics).map(([heuristic, metrics]) => (
                      <TableRow 
                        key={heuristic}
                        sx={{ 
                          bgcolor: heuristic === selectedHeuristic ? '#e3f2fd' : 'transparent',
                          fontWeight: heuristic === selectedHeuristic ? 'bold' : 'normal'
                        }}
                      >
                        <TableCell>{heuristic}</TableCell>
                        <TableCell align="right">{metrics.makespan?.toFixed(0) || 'N/A'}</TableCell>
                        <TableCell align="right">{metrics.avg_completion?.toFixed(0) || 'N/A'}</TableCell>
                        <TableCell align="right">{metrics.total_tardiness?.toFixed(0) || 'N/A'}</TableCell>
                        <TableCell align="right">{metrics.utilization?.toFixed(1) || 'N/A'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </Paper>
      )}

      {/* AI Insights Dialog */}
      <Dialog 
        open={aiInsightsOpen} 
        onClose={() => setAiInsightsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AIIcon color="primary" />
            <Typography variant="h6">AI Insights & Recommendations</Typography>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
            {aiInsights || 'Analyzing scheduling results...'}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAiInsightsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default ExcelUpload;
