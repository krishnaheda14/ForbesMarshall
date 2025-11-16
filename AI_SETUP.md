# AI Features Setup Guide

## 🤖 Gemini AI Integration

This CNC Scheduling System now includes AI-powered insights using Google's Gemini AI.

### Prerequisites

1. **Get a Gemini API Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Sign in with your Google account
   - Create a new API key
   - Copy the key

2. **Install Required Packages**
   ```bash
   pip install google-generativeai python-dotenv
   ```

3. **Configure Environment Variables**
   - Create a `.env` file in the project root directory
   - Add your API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```
   - Save the file

### AI Features Available

Once configured, you'll see **🤖 AI Insights** buttons throughout the application:

#### 1. **KPI Dashboard AI Analysis**
   - Location: Selected Heuristic View → KPI Dashboard tab
   - Button: "🤖 Get AI Insights on Performance"
   - Provides: Performance analysis, bottleneck identification, improvement recommendations

#### 2. **Heuristic Comparison AI Analysis**
   - Location: Heuristic Comparison page
   - Button: "🤖 AI Analysis"
   - Provides: Comparison rationale, trade-offs, scenario recommendations

#### 3. **Machine Breakdown Impact Analysis**
   - Location: Appears automatically when you simulate a breakdown
   - Section: "🤖 AI Breakdown Impact Analysis" (expandable)
   - Provides: Impact assessment, cost implications, risk mitigation strategies

#### 4. **Capacity Analysis AI Recommendation**
   - Location: Sidebar → Add Job → Analyze Capacity
   - Button: "🤖 Get AI Recommendation"
   - Provides: Feasibility validation, risk assessment, alternative strategies

#### 5. **Operation Status AI Insights**
   - Location: Selected Heuristic View → Operation Status tab
   - Button: "🤖 Get AI Insights on Operation Status"
   - Provides: Schedule health assessment, bottleneck identification, optimization suggestions

#### 6. **Activity Log Pattern Analysis**
   - Location: Heuristic Comparison → Activity Log section
   - Button: "🤖 Analyze Activity Patterns"
   - Provides: Usage patterns, operational inefficiencies, workflow recommendations

#### 7. **Gantt Chart Timeline Analysis**
   - Location: Selected Heuristic View → Gantt Chart tab
   - Button: "🤖 Get AI Insights on Schedule Visualization"
   - Provides: Load balancing analysis, scheduling bottlenecks, resource allocation tips

### Troubleshooting

**Issue: "AI insights are disabled" message**
- Solution: Check that your `.env` file exists and contains `GEMINI_API_KEY=your_key`

**Issue: "Error generating AI insights"**
- Solution 1: Verify your API key is valid and active
- Solution 2: Check your internet connection
- Solution 3: Ensure you haven't exceeded the API quota

**Issue: AI buttons not appearing**
- Solution: Restart the Streamlit application after adding the `.env` file

### Security Notes

- **Never commit** your `.env` file to version control
- The `.env` file is already in `.gitignore`
- Use `.env.example` as a template for team members
- Keep your API key confidential

### Cost Considerations

- Gemini API has a free tier with generous limits
- Each AI insight button click makes one API call
- Monitor your usage at: https://makersuite.google.com/

### Disabling AI Features

To disable AI features:
1. Remove or rename the `.env` file, OR
2. Remove the `GEMINI_API_KEY` from `.env`

The application will continue to work normally without AI insights.
