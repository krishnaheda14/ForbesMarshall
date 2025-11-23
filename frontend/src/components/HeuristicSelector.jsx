// src/components/HeuristicSelector.jsx
import React from 'react';
import {
  FormControl,
  Select,
  MenuItem,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import { Psychology as AIIcon } from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';

const heuristics = [
  { value: 'SPT', label: 'SPT', description: 'Shortest Processing Time' },
  { value: 'EDD', label: 'EDD', description: 'Earliest Due Date' },
  { value: 'CR', label: 'CR', description: 'Critical Ratio' },
  { value: 'PRIORITY', label: 'PRIORITY', description: 'Priority-Based' },
];

function HeuristicSelector() {
  const { currentHeuristic, setCurrentHeuristic } = useSchedulerStore();

  return (
    <Box>
      <Typography variant="caption" sx={{ mb: 1, display: 'block', opacity: 0.9 }}>
        <AIIcon sx={{ fontSize: 14, mr: 0.5, verticalAlign: 'middle' }} />
        Scheduling Algorithm
      </Typography>
      <FormControl fullWidth size="small">
        <Select
          value={currentHeuristic || ''}
          onChange={(e) => setCurrentHeuristic(e.target.value)}
          displayEmpty
          sx={{
            backgroundColor: 'rgba(255,255,255,0.15)',
            color: 'white',
            borderRadius: 2,
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(255,255,255,0.3)',
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(255,255,255,0.5)',
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              borderColor: 'white',
            },
            '& .MuiSvgIcon-root': {
              color: 'white',
            },
          }}
        >
          <MenuItem value="" disabled>
            <em>Select Heuristic</em>
          </MenuItem>
          {heuristics.map((h) => (
            <MenuItem key={h.value} value={h.value}>
              <Box>
                <Typography variant="body2" fontWeight="bold">
                  {h.label}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {h.description}
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {currentHeuristic && (
        <Chip
          label={`Active: ${currentHeuristic}`}
          size="small"
          color="success"
          sx={{ mt: 1, color: 'white' }}
        />
      )}
    </Box>
  );
}

export default HeuristicSelector;
