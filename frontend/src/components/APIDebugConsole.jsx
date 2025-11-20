// src/components/APIDebugConsole.jsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Paper,
  IconButton,
  Badge,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  BugReport as BugIcon,
  Clear as ClearIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

// Global store for API logs
window.apiDebugLogs = window.apiDebugLogs || [];

export const logAPICall = (logEntry) => {
  window.apiDebugLogs = window.apiDebugLogs || [];
  window.apiDebugLogs.unshift(logEntry);
  
  // Keep only last 50 logs
  if (window.apiDebugLogs.length > 50) {
    window.apiDebugLogs = window.apiDebugLogs.slice(0, 50);
  }
  
  // Trigger custom event for UI update
  window.dispatchEvent(new CustomEvent('apiDebugUpdate'));
};

function APIDebugConsole() {
  const [logs, setLogs] = useState([]);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    // Initial load
    setLogs([...(window.apiDebugLogs || [])]);

    // Listen for updates
    const handleUpdate = () => {
      setLogs([...(window.apiDebugLogs || [])]);
    };

    window.addEventListener('apiDebugUpdate', handleUpdate);
    return () => window.removeEventListener('apiDebugUpdate', handleUpdate);
  }, []);

  const handleClear = () => {
    window.apiDebugLogs = [];
    setLogs([]);
  };

  const getStatusColor = (status) => {
    if (status >= 200 && status < 300) return 'success';
    if (status >= 400) return 'error';
    return 'warning';
  };

  const getMethodColor = (method) => {
    const colors = {
      GET: 'info',
      POST: 'primary',
      PUT: 'warning',
      DELETE: 'error',
      OPTIONS: 'default',
    };
    return colors[method] || 'default';
  };

  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
    });
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        backgroundColor: '#1e1e1e',
        color: '#d4d4d4',
        borderTop: '2px solid #007acc',
        maxHeight: expanded ? '60vh' : '50px',
        transition: 'max-height 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 1,
          backgroundColor: '#252526',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid #3e3e42' : 'none',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <BugIcon sx={{ color: '#007acc', fontSize: 20 }} />
          <Typography variant="body2" fontWeight="bold">
            API Debug Console
          </Typography>
          <Badge badgeContent={logs.length} color="primary" max={99}>
            <Chip
              label={`${logs.length} calls`}
              size="small"
              sx={{ backgroundColor: '#3e3e42', color: '#d4d4d4' }}
            />
          </Badge>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              handleClear();
            }}
            sx={{ color: '#d4d4d4' }}
          >
            <ClearIcon fontSize="small" />
          </IconButton>
          <ExpandMoreIcon
            sx={{
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s',
              color: '#d4d4d4',
            }}
          />
        </Box>
      </Box>

      {/* Console Content */}
      {expanded && (
        <Box
          sx={{
            overflowY: 'auto',
            p: 2,
            flex: 1,
            fontFamily: 'Consolas, Monaco, monospace',
            fontSize: '12px',
          }}
        >
          {logs.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              No API calls logged yet. Make some requests to see them here.
            </Typography>
          ) : (
            logs.map((log, index) => (
              <Paper
                key={index}
                sx={{
                  mb: 1.5,
                  p: 1.5,
                  backgroundColor: '#2d2d30',
                  border: '1px solid #3e3e42',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Chip
                    label={log.method}
                    size="small"
                    color={getMethodColor(log.method)}
                    sx={{ minWidth: '60px', fontWeight: 'bold' }}
                  />
                  {log.status && (
                    <Chip
                      label={log.status}
                      size="small"
                      color={getStatusColor(log.status)}
                      sx={{ minWidth: '50px' }}
                    />
                  )}
                  <Chip
                    label={formatDuration(log.duration)}
                    size="small"
                    sx={{
                      backgroundColor: log.duration > 2000 ? '#f44336' : log.duration > 1000 ? '#ff9800' : '#4caf50',
                      color: 'white',
                    }}
                  />
                  <Typography variant="caption" sx={{ color: '#858585', ml: 'auto' }}>
                    {formatTime(log.timestamp)}
                  </Typography>
                </Box>

                <Typography variant="body2" sx={{ color: '#4ec9b0', mb: 0.5 }}>
                  {log.url}
                </Typography>

                {log.error && (
                  <Box sx={{ mt: 1, p: 1, backgroundColor: '#5a1d1d', borderRadius: 1 }}>
                    <Typography variant="caption" sx={{ color: '#f48771', fontWeight: 'bold' }}>
                      ❌ ERROR:
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#f48771', display: 'block', mt: 0.5 }}>
                      {log.error}
                    </Typography>
                  </Box>
                )}

                {log.request && (
                  <Accordion
                    sx={{
                      mt: 1,
                      backgroundColor: '#1e1e1e',
                      '&:before': { display: 'none' },
                    }}
                  >
                    <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#d4d4d4' }} />}>
                      <Typography variant="caption" sx={{ color: '#9cdcfe' }}>
                        📤 Request Body ({JSON.stringify(log.request).length} bytes)
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <pre
                        style={{
                          margin: 0,
                          color: '#ce9178',
                          fontSize: '11px',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {JSON.stringify(log.request, null, 2)}
                      </pre>
                    </AccordionDetails>
                  </Accordion>
                )}

                {log.response && (
                  <Accordion
                    sx={{
                      mt: 1,
                      backgroundColor: '#1e1e1e',
                      '&:before': { display: 'none' },
                    }}
                  >
                    <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#d4d4d4' }} />}>
                      <Typography variant="caption" sx={{ color: '#9cdcfe' }}>
                        📥 Response ({JSON.stringify(log.response).length} bytes)
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <pre
                        style={{
                          margin: 0,
                          color: '#b5cea8',
                          fontSize: '11px',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {JSON.stringify(log.response, null, 2)}
                      </pre>
                    </AccordionDetails>
                  </Accordion>
                )}
              </Paper>
            ))
          )}
        </Box>
      )}
    </Box>
  );
}

export default APIDebugConsole;
