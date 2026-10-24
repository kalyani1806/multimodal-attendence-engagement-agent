"""
EPIC 6: Interface, Testing & Deployment - UI Components and Styling
Custom CSS and visual cards for Streamlit Dashboard.
"""

import streamlit as st

def apply_custom_styles():
    """Applies modern clean EdTech portal stylesheet."""
    st.markdown(
        """
        <style>
        /* General page layout and fonts */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 15px;
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 4px;
        }
        .suggestion-card {
            background-color: #FFFFFF;
            border-left: 5px solid #2563EB;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        }
        .badge-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-high {
            background-color: #FEE2E2;
            color: #991B1B;
        }
        .badge-medium {
            background-color: #FEF3C7;
            color: #92400E;
        }
        .badge-positive {
            background-color: #D1FAE5;
            color: #065F46;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_metric_card(title: str, value: str, subtitle: str = "", delta_color: str = "#2563EB"):
    """Renders a responsive metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div style="font-size: 0.8rem; color: {delta_color}; margin-top: 4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_ai_suggestion_card(suggestion: dict):
    """Renders an AI pedagogical recommendation card."""
    priority = suggestion.get("priority", "INFO")
    badge_class = "badge-high" if priority == "HIGH" else ("badge-positive" if priority == "POSITIVE" else "badge-medium")
    category = suggestion.get("category", "Pedagogical Note")
    obs = suggestion.get("observation", "")
    act = suggestion.get("action", "")
    reason = suggestion.get("pedagogical_reason", "")
    timestamp = suggestion.get("timestamp", 0)

    st.markdown(
        f"""
        <div class="suggestion-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 1rem; color: #1E293B;">{category}</span>
                <span class="badge-pill {badge_class}">{priority} • {timestamp:.0f}s</span>
            </div>
            <p style="margin: 4px 0; font-size: 0.9rem; color: #475569;"><b>Observation:</b> {obs}</p>
            <p style="margin: 4px 0; font-size: 0.95rem; color: #1E40AF; font-weight: 600;"><b>Suggested Action:</b> {act}</p>
            <p style="margin: 4px 0; font-size: 0.85rem; color: #64748B; font-style: italic;"><b>AI Reasoning:</b> {reason}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
