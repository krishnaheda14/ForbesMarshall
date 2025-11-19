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
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  AutoFixHigh as AutoMapIcon,
  CheckCircle as ConfirmIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import axios from 'axios';

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
      
      const mappingData = response.data.mappings;
      setColumns(mappingData);
      
      // Set initial mappings
      const initialMappings = {};
      mappingData.forEach(m => {
        initialMappings[m.excel_column] = m.canonical_field;
      });
      setMappings(initialMappings);
      
      // Get available fields
      if (mappingData.length > 0) {
        setAvailableFields(mappingData[0].available_fields || []);
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
      
      enqueueSnackbar(
        `Scheduled ${response.data.job_count} jobs using ${heuristic}!`, 
        { variant: 'success' }
      );
      
      // Navigate to dashboard or Gantt view
      setTimeout(() => {
        window.location.href = '/';
      }, 1500);
      
    } catch (error) {
      enqueueSnackbar(`Scheduling failed: ${error.response?.data?.detail || error.message}`, { 
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

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        📊 Excel Data Import
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
                          {availableFields.map(field => (
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
              🚀 Schedule Jobs
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Choose a scheduling algorithm to compute optimal job sequences:
            </Typography>
            
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
              💡 Tip: Each algorithm optimizes for different criteria. SPT minimizes average completion time, 
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
              }}
              sx={{ mt: 2, alignSelf: 'flex-start' }}
            >
              Upload Another File
            </Button>
          </Box>
        </Paper>
      )}
    </Container>
  );
}

export default ExcelUpload;
