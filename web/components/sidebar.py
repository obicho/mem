import streamlit as st
from app.client import Memory


def render_sidebar(memory_client: Memory) -> tuple[str, str | None]:
    """Render sidebar with navigation and filters.

    Args:
        memory_client: The Memory client instance

    Returns:
        Tuple of (selected_page, user_id_filter)
    """
    with st.sidebar:
        st.title("Memory Manager")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            options=["Search", "Add Memory", "Browse All"],
            format_func=lambda x: {
                "Search": "Search",
                "Add Memory": "Add Memory",
                "Browse All": "Browse All",
            }[x],
            label_visibility="collapsed",
        )

        st.divider()

        # User ID filter
        user_id_filter = st.text_input(
            "Filter by User ID",
            placeholder="Enter user_id...",
            help="Filter memories by user ID",
        )

        # Memory count
        st.divider()
        try:
            if user_id_filter:
                count = memory_client.count(user_id=user_id_filter)
                st.metric("Memories", count, help=f"Memories for user: {user_id_filter}")
            else:
                count = memory_client.count()
                st.metric("Total Memories", count)
        except Exception as e:
            st.error(f"Error counting memories: {e}")

    return page, user_id_filter if user_id_filter else None
