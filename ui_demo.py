"""
Demo: Using Professional UI Components in CNC Scheduling App
Run this file to see the new UI components in action
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import custom components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.custom_ui import (
    render_hero_section,
    render_stat_cards_row,
    render_alert,
    render_progress_bar,
    render_badge,
    render_card,
    render_loading_spinner
)

# Page config
st.set_page_config(
    page_title="CNC Scheduler - Professional UI Demo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit branding for production look
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Hero Section
render_hero_section(
    title="CNC Job Scheduling System",
    subtitle="AI-powered optimization platform for manufacturing excellence. Real-time scheduling with 6+ heuristic algorithms.",
    icon="🏭",
    gradient="blue"
)

# Demo of all components
st.markdown("## 📊 Component Gallery")

# 1. Stat Cards
st.markdown("### 1️⃣ Professional KPI Cards")
render_stat_cards_row([
    {
        "label": "Makespan",
        "value": "14.2 days",
        "delta": "-12%",
        "icon": "⏱️",
        "delta_color": "green"
    },
    {
        "label": "Machine Utilization",
        "value": "87.3%",
        "delta": "+5.2%",
        "icon": "📊",
        "delta_color": "green"
    },
    {
        "label": "On-Time Delivery",
        "value": "94%",
        "delta": "+8%",
        "icon": "✅",
        "delta_color": "green"
    },
    {
        "label": "Total Cost",
        "value": "$45,230",
        "delta": "-$3.2K",
        "icon": "💰",
        "delta_color": "green"
    }
])

st.markdown("---")

# 2. Alerts
st.markdown("### 2️⃣ Alert Components")

col1, col2 = st.columns(2)

with col1:
    render_alert(
        "Schedule computed successfully! All operations scheduled within deadlines.",
        type="success",
        dismissible=True
    )
    
    render_alert(
        "Machine M3 will require maintenance in 2 days. Plan accordingly.",
        type="warning",
        dismissible=True
    )

with col2:
    render_alert(
        "New AI insights available. Check the recommendations tab.",
        type="info",
        dismissible=True
    )
    
    render_alert(
        "Failed to load vendor data. Using in-house processing only.",
        type="error",
        dismissible=True
    )

st.markdown("---")

# 3. Progress Bars
st.markdown("### 3️⃣ Progress Indicators")

col1, col2 = st.columns(2)

with col1:
    render_progress_bar(
        value=75,
        max_value=100,
        label="Schedule Computation Progress",
        color="#3b82f6",
        show_percentage=True
    )
    
    render_progress_bar(
        value=87.3,
        max_value=100,
        label="Machine Utilization",
        color="#10b981",
        show_percentage=True
    )

with col2:
    render_progress_bar(
        value=34,
        max_value=50,
        label="Operations Completed",
        color="#f59e0b",
        show_percentage=False
    )
    
    render_progress_bar(
        value=94,
        max_value=100,
        label="On-Time Delivery Rate",
        color="#8b5cf6",
        show_percentage=True
    )

st.markdown("---")

# 4. Badges
st.markdown("### 4️⃣ Status Badges")

st.markdown("**Operation Status:**")
badges_html = f"""
<div style='display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;'>
    {render_badge("SCHEDULED", "blue", "medium")}
    {render_badge("IN PROGRESS", "yellow", "medium")}
    {render_badge("COMPLETED", "green", "medium")}
    {render_badge("DELAYED", "red", "medium")}
    {render_badge("OUTSOURCED", "purple", "medium")}
    {render_badge("PENDING", "gray", "medium")}
</div>
"""
st.markdown(badges_html, unsafe_allow_html=True)

st.markdown("**Priority Levels:**")
priority_badges = f"""
<div style='display: flex; gap: 0.5rem; margin-bottom: 2rem;'>
    {render_badge("PRIORITY 1", "red", "small")}
    {render_badge("PRIORITY 2", "yellow", "small")}
    {render_badge("PRIORITY 3", "blue", "small")}
    {render_badge("PRIORITY 4", "gray", "small")}
</div>
"""
st.markdown(priority_badges, unsafe_allow_html=True)

st.markdown("---")

# 5. Cards
st.markdown("### 5️⃣ Card Components")

col1, col2, col3 = st.columns(3)

with col1:
    render_card(
        content="""
        <div style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🎯</div>
            <h3 style='color: #1e3a8a; margin-bottom: 0.5rem;'>SPT Algorithm</h3>
            <p style='color: #6b7280;'>Shortest Processing Time first. Minimizes makespan and average flow time.</p>
            <div style='margin-top: 1rem;'>
                """ + render_badge("RECOMMENDED", "green", "small") + """
            </div>
        </div>
        """,
        variant="default"
    )

with col2:
    render_card(
        content="""
        <div style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📅</div>
            <h3 style='color: #1e3a8a; margin-bottom: 0.5rem;'>EDD Algorithm</h3>
            <p style='color: #6b7280;'>Earliest Due Date. Prioritizes urgent jobs to minimize tardiness.</p>
            <div style='margin-top: 1rem;'>
                """ + render_badge("ALTERNATIVE", "blue", "small") + """
            </div>
        </div>
        """,
        variant="bordered"
    )

with col3:
    render_card(
        content="""
        <div style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>⚖️</div>
            <h3 style='color: #1e3a8a; margin-bottom: 0.5rem;'>CR Algorithm</h3>
            <p style='color: #6b7280;'>Critical Ratio. Balances urgency with work remaining.</p>
            <div style='margin-top: 1rem;'>
                """ + render_badge("BALANCED", "purple", "small") + """
            </div>
        </div>
        """,
        variant="shadow"
    )

st.markdown("---")

# 6. Card with Title and Footer
st.markdown("### 6️⃣ Full-Featured Card")

render_card(
    title="🤖 AI Recommendation",
    content="""
    <p style='line-height: 1.8; color: #374151;'>
        Based on your current job mix and constraints, the <strong>SPT (Shortest Processing Time)</strong> 
        algorithm is recommended. This will:
    </p>
    <ul style='line-height: 2; color: #374151;'>
        <li>✅ Reduce makespan by approximately 12%</li>
        <li>✅ Improve machine utilization to 87%+</li>
        <li>✅ Minimize average completion time</li>
        <li>⚠️ May increase tardiness for long jobs</li>
    </ul>
    <p style='line-height: 1.8; color: #374151; margin-top: 1rem;'>
        <strong>Alternative:</strong> Consider <strong>WEIGHTED</strong> algorithm if on-time delivery is critical.
    </p>
    """,
    footer="Generated by Gemini AI • Last updated: 2 minutes ago",
    variant="gradient"
)

st.markdown("---")

# 7. Loading Spinner Demo
st.markdown("### 7️⃣ Loading States")

if st.button("🧪 Simulate Schedule Computation"):
    render_loading_spinner("Computing optimal schedule with SPT algorithm...", size="large")
    import time
    time.sleep(2)
    st.success("✅ Schedule computed successfully!")
    st.balloons()

st.markdown("---")

# 8. Real-world Example
st.markdown("## 🎬 Real-World Usage Example")

render_hero_section(
    title="Heuristic Comparison Results",
    subtitle="All 6 algorithms analyzed. Best option highlighted below.",
    icon="⚖️",
    gradient="purple"
)

# Simulated comparison data
comparison_data = [
    {"label": "SPT Score", "value": "92/100", "delta": "+8", "icon": "🎯", "delta_color": "green"},
    {"label": "EDD Score", "value": "78/100", "delta": "-5", "icon": "📅", "delta_color": "red"},
    {"label": "CR Score", "value": "85/100", "delta": "+2", "icon": "⚖️", "delta_color": "green"},
    {"label": "WEIGHTED", "value": "88/100", "delta": "+4", "icon": "🎚️", "delta_color": "green"}
]

render_stat_cards_row(comparison_data)

render_alert(
    "🏆 SPT algorithm achieved the highest composite score. Click 'Apply SPT' to update your schedule.",
    type="success"
)

# Sidebar Demo
with st.sidebar:
    st.markdown("### 🎨 Professional Sidebar")
    
    st.markdown(render_badge("v2.0", "blue", "small"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### 🧮 Quick Actions")
    
    if st.button("🧪 Compute All", use_container_width=True):
        st.info("Computing all heuristics...")
    
    if st.button("✅ Apply Best", use_container_width=True):
        st.success("Applied SPT algorithm!")
    
    if st.button("📥 Export Results", use_container_width=True):
        st.success("Exported to CSV!")
    
    st.markdown("---")
    
    st.markdown("#### 📊 System Status")
    
    render_progress_bar(87.3, 100, "Machine Util.", "#10b981", True)
    render_progress_bar(94, 100, "On-Time %", "#3b82f6", True)
    render_progress_bar(34, 50, "Jobs Done", "#f59e0b", False)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p style='font-weight: 600; margin-bottom: 0.5rem;'>🏭 ForbesMarshall CNC Scheduling System</p>
    <p style='font-size: 0.875rem;'>Production-Ready UI Components Demo</p>
    <p style='font-size: 0.75rem; margin-top: 1rem;'>
        Made with ❤️ using Streamlit + Custom Components
    </p>
</div>
""", unsafe_allow_html=True)
