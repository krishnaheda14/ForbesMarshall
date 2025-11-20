// src/components/Sidebar.jsx
import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
  IconButton,
  Collapse,
  Chip,
  Alert,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  BarChart as ComparisonIcon,
  Timeline as GanttIcon,
  List as OperationsIcon,
  Settings as SettingsIcon,
  Factory as FactoryIcon,
  ExpandMore,
  ExpandLess,
  PlayArrow,
  CloudUpload as UploadIcon,
  AttachMoney as CostIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';

import HeuristicSelector from './HeuristicSelector';
import ComputeControls from './ComputeControls';
import MachineryControls from './MachineryControls';
import useSchedulerStore from '../store/useSchedulerStore';

const drawerWidth = 280;

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Excel Upload', icon: <UploadIcon />, path: '/excel-upload' },
  { text: 'Comparison', icon: <ComparisonIcon />, path: '/comparison' },
  { text: 'Gantt Chart', icon: <GanttIcon />, path: '/gantt' },
  { text: 'Operations', icon: <OperationsIcon />, path: '/operations' },
  { text: 'Cost Analysis', icon: <CostIcon />, path: '/cost-analysis' },
  { text: 'Outsourcing', icon: <FactoryIcon />, path: '/outsourcing' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [controlsOpen, setControlsOpen] = useState(true);
  const { dataLoaded, dataStats } = useSchedulerStore();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          background: 'linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%)',
          color: 'white',
        },
      }}
    >
      <Box sx={{ p: 3, pb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <FactoryIcon sx={{ fontSize: 32, mr: 1.5 }} />
          <Typography variant="h6" component="div" fontWeight="bold">
            CNC Scheduler
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ opacity: 0.8 }}>
          v2.0 - Advanced Scheduling
        </Typography>
        
        {/* Dataset Status Indicator */}
        <Box sx={{ mt: 2 }}>
          {dataLoaded ? (
            <Alert 
              severity="success" 
              icon={<CheckIcon />}
              sx={{ 
                py: 0.5, 
                fontSize: '0.75rem',
                bgcolor: 'rgba(76, 175, 80, 0.2)',
                color: 'white',
                '& .MuiAlert-icon': { color: 'white' }
              }}
            >
              <Box>
                <Typography variant="caption" fontWeight="bold" display="block">
                  Dataset Loaded
                </Typography>
                <Box display="flex" gap={0.5} mt={0.5} flexWrap="wrap">
                  <Chip 
                    label={`${dataStats?.operations || 0} Ops`} 
                    size="small" 
                    sx={{ height: 18, fontSize: '0.65rem', bgcolor: 'rgba(255,255,255,0.3)', color: 'white' }}
                  />
                  <Chip 
                    label={`${dataStats?.jobs || 0} Jobs`} 
                    size="small" 
                    sx={{ height: 18, fontSize: '0.65rem', bgcolor: 'rgba(255,255,255,0.3)', color: 'white' }}
                  />
                </Box>
              </Box>
            </Alert>
          ) : (
            <Alert 
              severity="warning" 
              icon={<WarningIcon />}
              sx={{ 
                py: 0.5, 
                fontSize: '0.75rem',
                bgcolor: 'rgba(255, 152, 0, 0.2)',
                color: 'white',
                '& .MuiAlert-icon': { color: 'white' }
              }}
            >
              <Typography variant="caption" fontWeight="bold">
                No Dataset Loaded
              </Typography>
            </Alert>
          )}
        </Box>
      </Box>

      <Divider sx={{ borderColor: 'rgba(255,255,255,0.12)' }} />

      <List sx={{ px: 1, py: 2 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => navigate(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  backgroundColor: 'rgba(255,255,255,0.15)',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.2)',
                  },
                },
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.08)',
                },
              }}
            >
              <ListItemIcon sx={{ color: 'white', minWidth: 40 }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{ fontWeight: 500 }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ borderColor: 'rgba(255,255,255,0.12)' }} />

      <Box sx={{ px: 2, py: 2 }}>
        <ListItemButton
          onClick={() => setControlsOpen(!controlsOpen)}
          sx={{
            borderRadius: 2,
            backgroundColor: 'rgba(255,255,255,0.1)',
            mb: 1,
          }}
        >
          <ListItemIcon sx={{ color: 'white', minWidth: 40 }}>
            <PlayArrow />
          </ListItemIcon>
          <ListItemText
            primary="Controls"
            primaryTypographyProps={{ fontWeight: 600 }}
          />
          {controlsOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>

        <Collapse in={controlsOpen} timeout="auto">
          <Box sx={{ mt: 2 }}>
            <HeuristicSelector />
            <Box sx={{ mt: 2 }}>
              <ComputeControls />
            </Box>
            <Box sx={{ mt: 2 }}>
              <MachineryControls />
            </Box>
          </Box>
        </Collapse>
      </Box>
    </Drawer>
  );
}

export default Sidebar;
