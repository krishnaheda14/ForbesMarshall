// src/components/AIInsightsPanel.jsx
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  IconButton,
  Paper,
  Chip,
  Divider,
} from '@mui/material';
import { 
  Close as CloseIcon, 
  Psychology as AIIcon,
  AutoAwesome as SparkleIcon,
  TrendingUp as TrendIcon,
} from '@mui/icons-material';

function AIInsightsPanel({ insights, onClose }) {
  // Parse insights to format nicely
  const formatInsights = (text) => {
    if (!text) return text;
    
    // Split by common markdown headers
    const sections = text.split(/(?=##)/);
    
    return sections.map((section, idx) => {
      // Check if it's a header section
      const headerMatch = section.match(/^##\s*(.+?)[\n\r]/);
      if (headerMatch) {
        const title = headerMatch[1].trim();
        const content = section.replace(/^##\s*.+?[\n\r]/, '').trim();
        
        return (
          <Box key={idx} sx={{ mb: idx < sections.length - 1 ? 3 : 0 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700, 
                mb: 1.5,
                color: '#1e3a8a',
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}
            >
              <SparkleIcon sx={{ fontSize: 20, color: '#3b82f6' }} />
              {title}
            </Typography>
            <Typography 
              variant="body1" 
              sx={{ 
                whiteSpace: 'pre-wrap', 
                lineHeight: 1.8,
                pl: 3.5,
                color: '#374151'
              }}
            >
              {formatContent(content)}
            </Typography>
          </Box>
        );
      }
      
      return (
        <Typography 
          key={idx}
          variant="body1" 
          sx={{ 
            whiteSpace: 'pre-wrap', 
            lineHeight: 1.8,
            mb: idx < sections.length - 1 ? 2 : 0,
            color: '#374151'
          }}
        >
          {formatContent(section)}
        </Typography>
      );
    });
  };

  const formatContent = (text) => {
    // Highlight important metrics with inline styling
    return text.split('\n').map((line, idx) => {
      // Check for bullet points or numbered lists
      if (line.match(/^[\s]*[-*]\s/)) {
        return (
          <Box key={idx} component="span" sx={{ display: 'block', ml: 1, mb: 0.5 }}>
            • {line.replace(/^[\s]*[-*]\s/, '')}
          </Box>
        );
      }
      if (line.match(/^[\s]*\d+\.\s/)) {
        return (
          <Box key={idx} component="span" sx={{ display: 'block', ml: 1, mb: 0.5, fontWeight: 500 }}>
            {line}
          </Box>
        );
      }
      return <span key={idx}>{line}<br /></span>;
    });
  };

  return (
    <Card
      elevation={8}
      sx={{
        mb: 3,
        background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #dbeafe 100%)',
        border: '2px solid',
        borderColor: '#3b82f6',
        borderRadius: 3,
        position: 'relative',
        overflow: 'visible',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899)',
        }
      }}
    >
      <CardContent sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          mb: 2 
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box
              sx={{
                width: 48,
                height: 48,
                borderRadius: 2,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)',
              }}
            >
              <AIIcon sx={{ fontSize: 28, color: 'white' }} />
            </Box>
            <Box>
              <Typography 
                variant="h5" 
                fontWeight="800"
                sx={{ 
                  background: 'linear-gradient(135deg, #1e3a8a, #3b82f6)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 0.5
                }}
              >
                AI-Powered Analysis
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip 
                  label="Powered by Gemini AI" 
                  size="small" 
                  sx={{ 
                    bgcolor: '#dbeafe',
                    color: '#1e40af',
                    fontWeight: 600,
                    fontSize: '0.7rem',
                    height: 20
                  }} 
                />
                <Chip 
                  icon={<TrendIcon sx={{ fontSize: 14 }} />}
                  label="Smart Insights" 
                  size="small" 
                  sx={{ 
                    bgcolor: '#e0e7ff',
                    color: '#4338ca',
                    fontWeight: 600,
                    fontSize: '0.7rem',
                    height: 20
                  }} 
                />
              </Box>
            </Box>
          </Box>
          <IconButton 
            onClick={onClose} 
            sx={{ 
              color: '#64748b',
              '&:hover': {
                bgcolor: 'rgba(0,0,0,0.05)',
                color: '#1e293b'
              }
            }}
          >
            <CloseIcon />
          </IconButton>
        </Box>

        <Divider sx={{ mb: 3, borderColor: '#cbd5e1' }} />

        {/* Content */}
        <Paper
          elevation={0}
          sx={{
            p: 3,
            backgroundColor: 'white',
            borderRadius: 2,
            border: '1px solid #e2e8f0',
            maxHeight: '600px',
            overflowY: 'auto',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: '#f1f5f9',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#cbd5e1',
              borderRadius: '4px',
              '&:hover': {
                background: '#94a3b8',
              },
            },
          }}
        >
          {formatInsights(insights)}
        </Paper>

        {/* Footer */}
        <Box sx={{ 
          mt: 2, 
          pt: 2, 
          borderTop: '1px dashed #cbd5e1',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1
        }}>
          <SparkleIcon sx={{ fontSize: 16, color: '#8b5cf6' }} />
          <Typography 
            variant="caption" 
            sx={{ 
              color: '#64748b',
              fontStyle: 'italic',
              fontSize: '0.75rem'
            }}
          >
            AI-generated recommendations based on your scheduling data
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}

export default AIInsightsPanel;
