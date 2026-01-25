import streamlit as st
from app.client import Memory
from web.components.memory_card import render_memory_card


def render_search(memory_client: Memory, user_id_filter: str | None = None) -> None:
    """Render the search interface.

    Args:
        memory_client: The Memory client instance
        user_id_filter: Optional user ID to filter results
    """
    st.header("Search Memories")

    # Search form
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter search query...",
            label_visibility="collapsed",
        )

    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Search options
    with st.expander("Search Options"):
        limit = st.slider("Number of results", min_value=1, max_value=50, value=10)

    # Perform search
    if query and (search_clicked or "last_query" in st.session_state):
        # Store query in session for persistence
        if search_clicked:
            st.session_state.last_query = query
            st.session_state.last_limit = limit
            st.session_state.last_user_filter = user_id_filter

        try:
            results = memory_client.search(
                query=st.session_state.get("last_query", query),
                user_id=st.session_state.get("last_user_filter", user_id_filter),
                limit=st.session_state.get("last_limit", limit),
            )

            if not results:
                st.info("No memories found matching your query.")
            else:
                st.subheader(f"Results ({len(results)})")

                for result in results:
                    render_search_result(memory_client, result)

        except Exception as e:
            st.error(f"Search failed: {e}")
    elif not query:
        st.info("Enter a search query to find memories.")


def render_search_result(memory_client: Memory, result: dict) -> None:
    """Render a single search result.

    Args:
        memory_client: The Memory client instance
        result: The search result dict containing id, content, score, metadata
    """
    memory_id = result.get("id", "")
    content = result.get("content", "")
    score = result.get("score", 0)
    metadata = result.get("metadata", {})

    with st.container(border=True):
        # Header with score and user info
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            st.metric("Score", f"{score:.2f}")

        with col2:
            user_id = metadata.get("user_id", "")
            agent_id = metadata.get("agent_id", "")
            badges = []
            if user_id:
                badges.append(f"User: {user_id}")
            if agent_id:
                badges.append(f"Agent: {agent_id}")
            if badges:
                st.caption(" | ".join(badges))

        with col3:
            if st.button("Delete", key=f"del_search_{memory_id}", type="secondary"):
                try:
                    memory_client.delete(memory_id)
                    st.toast("Memory deleted!", icon="")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

        # Content preview
        preview = content[:200] + "..." if len(content) > 200 else content
        st.write(preview)

        # Expandable full view
        with st.expander("View Full Details"):
            render_memory_card(memory_client, memory_id, content, metadata)
