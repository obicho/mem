from pathlib import Path

import streamlit as st
from mem.client import Memory
from web.components.memory_card import render_memory_card


def render_search(memory_client: Memory, user_id_filter: str | None = None) -> None:
    """Render the search interface with text and image search.

    Args:
        memory_client: The Memory client instance
        user_id_filter: Optional user ID to filter results
    """
    st.header("Search Memories")

    # Tabs for different search modes
    tab_text, tab_image = st.tabs(["Text Search", "Image Search"])

    with tab_text:
        render_text_search(memory_client, user_id_filter)

    with tab_image:
        render_image_search(memory_client, user_id_filter)


def render_text_search(memory_client: Memory, user_id_filter: str | None = None) -> None:
    """Render text-based search interface."""
    # Search form
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter search query...",
            label_visibility="collapsed",
            key="text_search_query",
        )

    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True, key="text_search_btn")

    # Search options
    with st.expander("Search Options"):
        limit = st.slider("Number of results", min_value=1, max_value=50, value=10, key="text_search_limit")

    # Perform search
    if query and (search_clicked or "last_text_query" in st.session_state):
        # Store query in session for persistence
        if search_clicked:
            st.session_state.last_text_query = query
            st.session_state.last_text_limit = limit
            st.session_state.last_text_user_filter = user_id_filter

        try:
            results = memory_client.search(
                query=st.session_state.get("last_text_query", query),
                user_id=st.session_state.get("last_text_user_filter", user_id_filter),
                limit=st.session_state.get("last_text_limit", limit),
            )

            if not results:
                st.info("No memories found matching your query.")
            else:
                st.subheader(f"Results ({len(results)})")
                search_query = st.session_state.get("last_text_query", query)

                for result in results:
                    render_search_result(memory_client, result, query=search_query, prefix="text")

        except Exception as e:
            st.error(f"Search failed: {e}")
    elif not query:
        st.info("Enter a search query to find memories.")


def render_image_search(memory_client: Memory, user_id_filter: str | None = None) -> None:
    """Render image-based search interface."""
    st.write("Upload an image to find similar images in memory.")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Query Image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        help="Upload an image to search for similar images.",
        key="image_search_upload",
    )

    # Search options
    with st.expander("Search Options"):
        limit = st.slider("Number of results", min_value=1, max_value=50, value=10, key="image_search_limit")

    # Show uploaded image preview
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Query Image", width=200)

        with col2:
            search_clicked = st.button(
                "Search Similar Images",
                type="primary",
                use_container_width=True,
                key="image_search_btn",
            )

            if search_clicked:
                try:
                    with st.spinner("Analyzing image and searching..."):
                        # Read image bytes
                        image_bytes = uploaded_file.read()
                        uploaded_file.seek(0)  # Reset for potential re-read

                        results = memory_client.search_image(
                            image_bytes=image_bytes,
                            user_id=user_id_filter,
                            limit=limit,
                        )

                    if not results:
                        st.info("No similar images found.")
                    else:
                        st.session_state.image_search_results = results
                        st.rerun()

                except Exception as e:
                    st.error(f"Image search failed: {e}")

    # Display results if available
    if "image_search_results" in st.session_state and st.session_state.image_search_results:
        results = st.session_state.image_search_results
        st.subheader(f"Similar Images ({len(results)})")

        for result in results:
            # Use "image_search" as query placeholder since actual caption isn't stored
            render_search_result(memory_client, result, query="[image_search]", prefix="image")


def render_search_result(
    memory_client: Memory,
    result: dict,
    query: str = "",
    prefix: str = "",
) -> None:
    """Render a single search result with feedback buttons.

    Args:
        memory_client: The Memory client instance
        result: The search result dict containing id, content, score, metadata
        query: The search query (for feedback)
        prefix: Prefix for unique keys
    """
    memory_id = result.get("id", "")
    content = result.get("content", "")
    score = result.get("score", 0)
    metadata = result.get("metadata", {})
    content_type = metadata.get("content_type", "")
    image_path = metadata.get("image_path", "")

    with st.container(border=True):
        # Header with score and user info
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

        with col1:
            st.metric("Score", f"{score:.2f}")

        with col2:
            user_id = metadata.get("user_id", "")
            agent_id = metadata.get("agent_id", "")
            badges = []
            if content_type:
                badges.append(f"Type: {content_type}")
            if user_id:
                badges.append(f"User: {user_id}")
            if agent_id:
                badges.append(f"Agent: {agent_id}")
            if badges:
                st.caption(" | ".join(badges))

        with col3:
            # Feedback buttons
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                if st.button("👍", key=f"fb_pos_{prefix}_{memory_id}", help="Helpful result"):
                    try:
                        memory_client.feedback(query=query, memory_id=memory_id, signal="positive")
                        st.toast("Thanks for the feedback!", icon="👍")
                    except Exception as e:
                        st.error(f"Failed to record feedback: {e}")
            with fb_col2:
                if st.button("👎", key=f"fb_neg_{prefix}_{memory_id}", help="Not helpful"):
                    try:
                        memory_client.feedback(query=query, memory_id=memory_id, signal="negative")
                        st.toast("Thanks for the feedback!", icon="👎")
                    except Exception as e:
                        st.error(f"Failed to record feedback: {e}")

        with col4:
            if st.button("Delete", key=f"del_{prefix}_{memory_id}", type="secondary"):
                try:
                    memory_client.delete(memory_id)
                    # Clear image search results if deleting from image search
                    if "image_search_results" in st.session_state:
                        del st.session_state.image_search_results
                    st.toast("Memory deleted!", icon="🗑️")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

        # Show image if this is an image memory
        if content_type == "image" and image_path and Path(image_path).exists():
            st.image(image_path, width=300)

        # Content preview (caption for images, text for others)
        is_chat_outcome = content_type == "chat_extract" and metadata.get("extract_category") == "outcome"
        if is_chat_outcome:
            st.markdown(f"**Outcome:** {content}")
        else:
            preview = content[:200] + "..." if len(content) > 200 else content
            st.write(preview)

        # Related memories for collapsed chat results
        related_memories = result.get("related_memories")
        if related_memories:
            with st.expander(f"Related memories ({len(related_memories)})"):
                for rel in related_memories:
                    rel_id = rel["id"]
                    rel_type = rel.get("content_type", "")
                    rel_cat = rel.get("extract_category", "")
                    label = rel_cat if rel_cat else rel_type
                    rel_mem = memory_client.get(rel_id)
                    if rel_mem:
                        rel_preview = rel_mem["content"][:120] + "..." if len(rel_mem["content"]) > 120 else rel_mem["content"]
                        st.markdown(f"- **{label}**: {rel_preview}  \n`{rel_id}`")
                    else:
                        st.markdown(f"- **{label}**: `{rel_id}`")

        # Expandable full view
        with st.expander("View Full Details"):
            render_memory_card(memory_client, memory_id, content, metadata)
