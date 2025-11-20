// src/components/SchedulingAnimation.jsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Chip,
  Grid,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Replay as ReplayIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

function SchedulingAnimation({ heuristic, schedule, metrics }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500); // ms per step

  const sortedSchedule = [...schedule].sort((a, b) => a.Start_Time - b.Start_Time);
  const totalSteps = sortedSchedule.length;

  useEffect(() => {
    let interval;
    if (isPlaying && currentStep < totalSteps) {
      interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, speed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentStep, totalSteps, speed]);

  const handlePlay = () => setIsPlaying(true);
  const handlePause = () => setIsPlaying(false);
  const handleReplay = () => {
    setCurrentStep(0);
    setIsPlaying(true);
  };

  const visibleSchedule = sortedSchedule.slice(0, currentStep + 1);
  const currentOp = sortedSchedule[currentStep];

  // Generate colors for jobs
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

  // Create animated Gantt chart
  const createGanttData = () => {
    return visibleSchedule.map((item) => ({
      x: [item.Start_Time, item.End_Time],
      y: [item.Machine_ID, item.Machine_ID],
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 25,
        color: getJobColor(item.Job_ID),
      },
      name: `${item.Job_ID}`,
      text: `${item.Job_ID}`,
      hovertemplate:
        `<b>Machine:</b> ${item.Machine_ID}<br>` +
        `<b>Job:</b> ${item.Job_ID}<br>` +
        `<b>Operation:</b> ${item.Operation_ID}<br>` +
        `<b>Start:</b> ${item.Start_Time.toFixed(0)} min<br>` +
        `<b>End:</b> ${item.End_Time.toFixed(0)} min<br>` +
        `<b>Duration:</b> ${(item.End_Time - item.Start_Time).toFixed(0)} min<br>` +
        `<extra></extra>`,
      hoverlabel: {
        bgcolor: 'white',
        font: { size: 12, color: 'black' },
      },
    }));
  };

  const progress = totalSteps > 0 ? ((currentStep + 1) / totalSteps) * 100 : 0;

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            {heuristic} Scheduling Animation
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Watch how {heuristic} schedules operations step-by-step
          </Typography>
        </Box>

        {/* Controls */}
        <Box sx={{ mb: 2, display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            size="small"
            startIcon={<PlayIcon />}
            onClick={handlePlay}
            disabled={isPlaying || currentStep >= totalSteps - 1}
          >
            Play
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<PauseIcon />}
            onClick={handlePause}
            disabled={!isPlaying}
          >
            Pause
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<ReplayIcon />}
            onClick={handleReplay}
          >
            Replay
          </Button>
          <Chip
            label={`Step: ${currentStep + 1} / ${totalSteps}`}
            color="primary"
            size="small"
          />
          <Chip
            label={`Speed: ${speed}ms`}
            size="small"
            onClick={() => setSpeed(speed === 500 ? 200 : speed === 200 ? 100 : 500)}
            sx={{ cursor: 'pointer' }}
          />
        </Box>

        {/* Progress Bar */}
        <Box sx={{ mb: 2 }}>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{ height: 8, borderRadius: 4 }}
          />
          <Typography variant="caption" color="text.secondary">
            {progress.toFixed(1)}% Complete
          </Typography>
        </Box>

        {/* Current Operation Info */}
        {currentOp && (
          <Box sx={{ mb: 2, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              <strong>Currently Scheduling:</strong>
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Job
                </Typography>
                <Typography variant="body2">
                  <strong>{currentOp.Job_ID}</strong>
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Operation
                </Typography>
                <Typography variant="body2">{currentOp.Operation_ID}</Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Machine
                </Typography>
                <Typography variant="body2">
                  <strong>{currentOp.Machine_ID}</strong>
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Time Slot
                </Typography>
                <Typography variant="body2">
                  {currentOp.Start_Time.toFixed(0)} - {currentOp.End_Time.toFixed(0)} min
                </Typography>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Animated Gantt Chart */}
        {visibleSchedule.length > 0 && (
          <Plot
            data={createGanttData()}
            layout={{
              title: `${heuristic} Schedule Progress`,
              xaxis: {
                title: 'Time (minutes)',
                showgrid: true,
                zeroline: false,
              },
              yaxis: {
                title: 'Machine',
                autorange: 'reversed',
              },
              height: 400,
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
        )}

        {/* Final Metrics Summary */}
        {currentStep >= totalSteps - 1 && metrics && (
          <Box sx={{ mt: 2, p: 2, bgcolor: '#e3f2fd', borderRadius: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              <strong>Final Results:</strong>
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Makespan
                </Typography>
                <Typography variant="h6">{metrics.Makespan_Days?.toFixed(2)} days</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Tardiness
                </Typography>
                <Typography variant="h6">{metrics.Total_Tardiness_Days?.toFixed(2)} days</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  On-Time %
                </Typography>
                <Typography variant="h6">{metrics['On_Time_%']?.toFixed(1)}%</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Utilization
                </Typography>
                <Typography variant="h6">{metrics['Machine_Utilization_%']?.toFixed(1)}%</Typography>
              </Grid>
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default SchedulingAnimation;
