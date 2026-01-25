"""Memory Manager Web App - Streamlit UI for managing AI agent memories."""

import os
import streamlit as st

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.client import Memory
from app.config import get_settings
from web.components.sidebar import render_sidebar
from web.components.add_memory import render_add_memory
from web.components.search import render_search
from web.components.memory_list import render_memory_list


def init_memory_client() -> Memory:
    """Initialize the Memory client, storing it in session state.

    Returns:
        The Memory client instance
    """
    if "memory_client" not in st.session_state:
        settings = get_settings()
        st.session_state.memory_client = Memory(api_key=settings.openai_api_key)

    return st.session_state.memory_client


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Memory Manager",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize memory client
    memory_client = init_memory_client()

    # Render sidebar and get navigation state
    page, user_id_filter = render_sidebar(memory_client)

    # Route to appropriate view
    if page == "Search":
        render_search(memory_client, user_id_filter)
    elif page == "Add Memory":
        render_add_memory(memory_client)
    elif page == "Browse All":
        render_memory_list(memory_client, user_id_filter)


if __name__ == "__main__":
    main()
