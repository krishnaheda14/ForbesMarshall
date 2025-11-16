# 🚀 Quick Start: AI Features

## Step 1: Install Required Packages
```bash
pip install google-generativeai python-dotenv
```

## Step 2: Get Your Gemini API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

## Step 3: Create .env File
Create a file named `.env` in the project folder:
```
GEMINI_API_KEY=paste_your_actual_key_here
```

## Step 4: Restart Streamlit
```bash
streamlit run cnc-scheduling.py
```

## Step 5: Look for 🤖 Buttons
You'll now see "🤖 Get AI Insights" buttons in:
- ✅ KPI Dashboard
- ✅ Heuristic Comparison
- ✅ Breakdown Simulator
- ✅ Add Job Analysis
- ✅ Operation Status
- ✅ Activity Log
- ✅ Gantt Chart

## That's It! 🎉
Click any 🤖 button to get expert AI insights!

---

**Note**: If you don't see the buttons, check:
1. `.env` file exists in the correct folder
2. API key is correct
3. Streamlit was restarted after creating `.env`
