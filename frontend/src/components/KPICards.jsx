// src/components/KPICards.jsx
import React from 'react';
import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import {
  Timer as TimerIcon,
  Warning as WarningIcon,
  AttachMoney as CostIcon,
  CheckCircle as OnTimeIcon,
  Memory as UtilizationIcon,
  CloudQueue as OutsourceIcon,
} from '@mui/icons-material';
import useSchedulerStore from '../store/useSchedulerStore';

function KPICards() {
  const { currentHeuristic, metrics } = useSchedulerStore();

  if (!currentHeuristic || !metrics[currentHeuristic]) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography color="text.secondary">
          No metrics available. Please compute heuristics first.
        </Typography>
      </Box>
    );
  }

  const currentMetrics = metrics[currentHeuristic]?.metrics || {};

  const kpiData = [
    {
      title: 'Makespan',
      value: `${currentMetrics.Makespan_Days || 0} days`,
      icon: <TimerIcon />,
      color: '#3b82f6',
    },
    {
      title: 'Total Tardiness',
      value: `${currentMetrics.Total_Tardiness_Days || 0} days`,
      icon: <WarningIcon />,
      color: '#f59e0b',
    },
    {
      title: 'Total Cost',
      value: `$${currentMetrics['Total_Cost_$'] || 0}`,
      icon: <CostIcon />,
      color: '#10b981',
    },
    {
      title: 'On-Time Delivery',
      value: `${currentMetrics['On_Time_%'] || 0}%`,
      icon: <OnTimeIcon />,
      color: '#059669',
    },
    {
      title: 'Machine Utilization',
      value: `${currentMetrics['Machine_Utilization_%'] || 0}%`,
      icon: <UtilizationIcon />,
      color: '#8b5cf6',
    },
    {
      title: 'Late Operations',
      value: `${currentMetrics.Late_Operations || 0} / ${currentMetrics.Total_Operations || 0}`,
      icon: <OutsourceIcon />,
      color: '#ef4444',
    }
  ];

  return (
    <Grid container spacing={3}>
      {kpiData.map((kpi, index) => (
        <Grid item xs={12} sm={6} md={4} key={index}>
          <Card
            sx={{
              height: '100%',
              background: `linear-gradient(135deg, ${kpi.color}15 0%, ${kpi.color}05 100%)`,
              borderLeft: `4px solid ${kpi.color}`,
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Box
                  sx={{
                    backgroundColor: kpi.color,
                    borderRadius: '50%',
                    p: 1,
                    mr: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                  }}
                >
                  {kpi.icon}
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {kpi.title}
                </Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold" sx={{ color: kpi.color }}>
                {kpi.value}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

export default KPICards;
