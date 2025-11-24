// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { SnackbarProvider } from 'notistack';

import Sidebar from './components/Sidebar';
import APIDebugConsole from './components/APIDebugConsole';
import Dashboard from './pages/Dashboard';
import Comparison from './pages/Comparison';
import GanttView from './pages/GanttView';
import OperationStatus from './pages/OperationStatus';
import Settings from './pages/Settings';
import ExcelUpload from './pages/ExcelUpload';
import CostAnalysis from './pages/CostAnalysis';
import OutsourcingAnalysis from './pages/OutsourcingAnalysis';
import ActivityLog from './pages/ActivityLog';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1e3a8a',
      light: '#3b82f6',
      dark: '#1e40af',
    },
    secondary: {
      main: '#10b981',
      light: '#34d399',
      dark: '#059669',
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      fontSize: '2.5rem',
    },
    h2: {
      fontWeight: 700,
      fontSize: '2rem',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        },
      },
    },
  },
});

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('❌ Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '40px', fontFamily: 'Arial' }}>
          <h1 style={{ color: 'red' }}>⚠️ Something went wrong</h1>
          <pre style={{ background: '#f5f5f5', padding: '20px', overflow: 'auto' }}>
            {this.state.error?.toString()}
            {'\n\n'}
            {this.state.error?.stack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <SnackbarProvider maxSnack={3} anchorOrigin={{ vertical: 'top', horizontal: 'right' }}>
          <Router>
            <Box sx={{ display: 'flex', minHeight: '100vh' }}>
              <Sidebar />
              <Box component="main" sx={{ flexGrow: 1, p: 3, backgroundColor: '#f8f9fa' }}>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/excel-upload" element={<ExcelUpload />} />
                  <Route path="/comparison" element={<Comparison />} />
                  <Route path="/gantt" element={<GanttView />} />
                  <Route path="/operations" element={<OperationStatus />} />
                  <Route path="/cost-analysis" element={<CostAnalysis />} />
                  <Route path="/outsourcing" element={<OutsourcingAnalysis />} />
                  <Route path="/activity-log" element={<ActivityLog />} />
                  <Route path="/settings" element={<Settings />} />
                </Routes>
              </Box>
            </Box>
            <APIDebugConsole />
          </Router>
        </SnackbarProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
