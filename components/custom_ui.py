"""
Custom UI Components for Production-Ready Interface
Provides React-like components using Streamlit + HTML/CSS/JS
"""

import streamlit as st
import streamlit.components.v1 as components

def render_hero_section(title, subtitle, icon="🏭", gradient="blue"):
    """
    Professional hero section with gradient background
    
    Args:
        title: Main heading text
        subtitle: Subtitle/description
        icon: Emoji icon
        gradient: Color theme (blue, purple, pink, green)
    """
    gradients = {
        "blue": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "purple": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "pink": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        "green": "linear-gradient(135deg, #0cebeb 0%, #20e3b2 100%)",
        "orange": "linear-gradient(135deg, #f83600 0%, #f9d423 100%)"
    }
    
    gradient_css = gradients.get(gradient, gradients["blue"])
    
    html = f"""
    <div style='
        background: {gradient_css};
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        animation: fadeIn 0.6s ease-in;
    '>
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
            <span style='font-size: 3rem;'>{icon}</span>
            <h1 style='color: white; margin: 0; border: none; padding: 0; font-size: 2.5rem; font-weight: 800;'>
                {title}
            </h1>
        </div>
        <p style='color: rgba(255,255,255,0.95); margin: 0; font-size: 1.2rem; line-height: 1.6;'>
            {subtitle}
        </p>
    </div>
    
    <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_stat_card(label, value, delta=None, delta_color="green", icon="📊"):
    """
    Professional stat card with optional delta indicator
    
    Args:
        label: Metric name
        value: Primary value to display
        delta: Change value (optional)
        delta_color: Color for delta (green/red/blue)
        icon: Icon emoji
    """
    delta_colors = {
        "green": "#10b981",
        "red": "#ef4444",
        "blue": "#3b82f6",
        "orange": "#f59e0b"
    }
    
    delta_html = ""
    if delta:
        color = delta_colors.get(delta_color, delta_colors["green"])
        arrow = "↑" if isinstance(delta, str) and not delta.startswith("-") else "↓"
        delta_html = f"""
        <div style='
            color: {color};
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        '>
            <span>{arrow}</span>
            <span>{delta}</span>
        </div>
        """
    
    html = f"""
    <div style='
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #3b82f6;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    ' onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 8px 30px rgba(0,0,0,0.12)';"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 20px rgba(0,0,0,0.08)';">
        
        <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.5rem;'>{icon}</span>
            <div style='color: #6b7280; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>
                {label}
            </div>
        </div>
        
        <div style='font-size: 2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.25rem;'>
            {value}
        </div>
        
        {delta_html}
    </div>
    """
    
    return html


def render_stat_cards_row(cards_data):
    """
    Render multiple stat cards in a responsive row
    
    Args:
        cards_data: List of dicts with keys: label, value, delta, icon, delta_color
    """
    num_cards = len(cards_data)
    cols = st.columns(num_cards)
    
    for col, card in zip(cols, cards_data):
        with col:
            html = render_stat_card(
                label=card.get('label', 'Metric'),
                value=card.get('value', '0'),
                delta=card.get('delta'),
                delta_color=card.get('delta_color', 'green'),
                icon=card.get('icon', '📊')
            )
            st.markdown(html, unsafe_allow_html=True)


def render_action_button(label, variant="primary", icon=None, full_width=False):
    """
    Professional action button with variants
    
    Args:
        label: Button text
        variant: Style variant (primary, secondary, success, danger, outline)
        icon: Optional icon emoji
        full_width: Whether to span full width
    
    Returns:
        Streamlit button component
    """
    variants = {
        "primary": {"bg": "#3b82f6", "hover": "#2563eb", "text": "white"},
        "secondary": {"bg": "#6b7280", "hover": "#4b5563", "text": "white"},
        "success": {"bg": "#10b981", "hover": "#059669", "text": "white"},
        "danger": {"bg": "#ef4444", "hover": "#dc2626", "text": "white"},
        "outline": {"bg": "transparent", "hover": "#f3f4f6", "text": "#3b82f6"}
    }
    
    style = variants.get(variant, variants["primary"])
    border = "2px solid #3b82f6" if variant == "outline" else "none"
    width = "100%" if full_width else "auto"
    
    button_text = f"{icon} {label}" if icon else label
    
    # Custom CSS for this button
    st.markdown(f"""
    <style>
        .custom-button-{variant} {{
            background: {style['bg']};
            color: {style['text']};
            border: {border};
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: {width};
            text-align: center;
            display: inline-block;
        }}
        
        .custom-button-{variant}:hover {{
            background: {style['hover']};
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
    </style>
    """, unsafe_allow_html=True)
    
    return st.button(button_text, use_container_width=full_width)


def render_alert(message, type="info", icon=None, dismissible=False):
    """
    Professional alert/notification component
    
    Args:
        message: Alert message text
        type: Alert type (info, success, warning, error)
        icon: Optional custom icon
        dismissible: Whether alert can be dismissed
    """
    alert_configs = {
        "info": {"bg": "#dbeafe", "border": "#3b82f6", "icon": "ℹ️", "text": "#1e40af"},
        "success": {"bg": "#d1fae5", "border": "#10b981", "icon": "✅", "text": "#065f46"},
        "warning": {"bg": "#fef3c7", "border": "#f59e0b", "icon": "⚠️", "text": "#92400e"},
        "error": {"bg": "#fee2e2", "border": "#ef4444", "icon": "❌", "text": "#991b1b"}
    }
    
    config = alert_configs.get(type, alert_configs["info"])
    display_icon = icon or config["icon"]
    
    dismiss_button = ""
    if dismissible:
        dismiss_button = """
        <button onclick="this.parentElement.style.display='none'" style='
            background: none;
            border: none;
            font-size: 1.25rem;
            cursor: pointer;
            color: inherit;
            opacity: 0.6;
            transition: opacity 0.2s;
        ' onmouseover='this.style.opacity="1"' onmouseout='this.style.opacity="0.6"'>
            ×
        </button>
        """
    
    html = f"""
    <div style='
        background-color: {config["bg"]};
        border-left: 4px solid {config["border"]};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        animation: slideIn 0.3s ease-out;
    '>
        <div style='display: flex; align-items: center; gap: 0.75rem;'>
            <span style='font-size: 1.25rem;'>{display_icon}</span>
            <span style='color: {config["text"]}; font-weight: 500; line-height: 1.5;'>
                {message}
            </span>
        </div>
        {dismiss_button}
    </div>
    
    <style>
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-20px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
    </style>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_progress_bar(value, max_value=100, label=None, color="#3b82f6", show_percentage=True):
    """
    Professional animated progress bar
    
    Args:
        value: Current progress value
        max_value: Maximum value (default 100)
        label: Optional label text
        color: Bar color
        show_percentage: Whether to show percentage text
    """
    percentage = (value / max_value) * 100
    
    label_html = ""
    if label:
        label_html = f"""
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
            <span style='font-weight: 600; color: #374151;'>{label}</span>
            <span style='font-weight: 600; color: {color};'>{percentage:.1f}%</span>
        </div>
        """ if show_percentage else f"""
        <div style='margin-bottom: 0.5rem; font-weight: 600; color: #374151;'>{label}</div>
        """
    
    html = f"""
    <div style='margin: 1rem 0;'>
        {label_html}
        <div style='
            background-color: #e5e7eb;
            border-radius: 9999px;
            height: 12px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
        '>
            <div style='
                background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
                height: 100%;
                width: {percentage}%;
                border-radius: 9999px;
                transition: width 0.8s ease-out;
                animation: progressGrow 0.8s ease-out;
                box-shadow: 0 0 10px {color}66;
            '></div>
        </div>
    </div>
    
    <style>
        @keyframes progressGrow {{
            from {{ width: 0%; }}
            to {{ width: {percentage}%; }}
        }}
    </style>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_badge(text, color="blue", size="medium"):
    """
    Professional badge/tag component
    
    Args:
        text: Badge text
        color: Color theme (blue, green, red, yellow, gray, purple)
        size: Size variant (small, medium, large)
    
    Returns:
        HTML string for badge
    """
    colors = {
        "blue": {"bg": "#dbeafe", "text": "#1e40af"},
        "green": {"bg": "#d1fae5", "text": "#065f46"},
        "red": {"bg": "#fee2e2", "text": "#991b1b"},
        "yellow": {"bg": "#fef3c7", "text": "#92400e"},
        "gray": {"bg": "#f3f4f6", "text": "#374151"},
        "purple": {"bg": "#f3e8ff", "text": "#6b21a8"}
    }
    
    sizes = {
        "small": {"padding": "0.25rem 0.5rem", "font": "0.75rem"},
        "medium": {"padding": "0.375rem 0.75rem", "font": "0.875rem"},
        "large": {"padding": "0.5rem 1rem", "font": "1rem"}
    }
    
    color_config = colors.get(color, colors["blue"])
    size_config = sizes.get(size, sizes["medium"])
    
    return f"""
    <span style='
        background-color: {color_config["bg"]};
        color: {color_config["text"]};
        padding: {size_config["padding"]};
        border-radius: 9999px;
        font-size: {size_config["font"]};
        font-weight: 600;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    '>
        {text}
    </span>
    """


def render_card(content, title=None, footer=None, variant="default"):
    """
    Professional card component
    
    Args:
        content: Main content (can be HTML or text)
        title: Optional card title
        footer: Optional footer content
        variant: Style variant (default, bordered, shadow, gradient)
    """
    variants = {
        "default": {
            "bg": "white",
            "shadow": "0 4px 20px rgba(0,0,0,0.08)",
            "border": "none"
        },
        "bordered": {
            "bg": "white",
            "shadow": "none",
            "border": "2px solid #e5e7eb"
        },
        "shadow": {
            "bg": "white",
            "shadow": "0 10px 40px rgba(0,0,0,0.12)",
            "border": "none"
        },
        "gradient": {
            "bg": "linear-gradient(135deg, #667eea22 0%, #764ba222 100%)",
            "shadow": "0 4px 20px rgba(0,0,0,0.08)",
            "border": "none"
        }
    }
    
    style = variants.get(variant, variants["default"])
    
    title_html = ""
    if title:
        title_html = f"""
        <div style='
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid #e5e7eb;
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e3a8a;
        '>
            {title}
        </div>
        """
    
    footer_html = ""
    if footer:
        footer_html = f"""
        <div style='
            padding: 1rem 1.5rem;
            border-top: 1px solid #e5e7eb;
            background-color: #f9fafb;
            font-size: 0.875rem;
            color: #6b7280;
        '>
            {footer}
        </div>
        """
    
    html = f"""
    <div style='
        background: {style["bg"]};
        border-radius: 12px;
        box-shadow: {style["shadow"]};
        border: {style["border"]};
        margin: 1rem 0;
        overflow: hidden;
        transition: transform 0.2s ease;
    ' onmouseover="this.style.transform='translateY(-2px)'"
       onmouseout="this.style.transform='translateY(0)'">
        {title_html}
        <div style='padding: 1.5rem;'>
            {content}
        </div>
        {footer_html}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_loading_spinner(text="Loading...", size="medium"):
    """
    Professional loading spinner
    
    Args:
        text: Loading text
        size: Spinner size (small, medium, large)
    """
    sizes = {
        "small": "30px",
        "medium": "50px",
        "large": "70px"
    }
    
    spinner_size = sizes.get(size, sizes["medium"])
    
    html = f"""
    <div style='
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        gap: 1rem;
    '>
        <div class='spinner' style='
            border: 4px solid #e5e7eb;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            width: {spinner_size};
            height: {spinner_size};
            animation: spin 1s linear infinite;
        '></div>
        <div style='color: #6b7280; font-weight: 500; font-size: 1.125rem;'>
            {text}
        </div>
    </div>
    
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    
    st.markdown(html, unsafe_allow_html=True)
