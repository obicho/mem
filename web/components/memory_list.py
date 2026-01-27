from pathlib import Path

import streamlit as st
from app.client import Memory
from web.components.memory_card import render_memory_card


def render_memory_list(memory_client: Memory, user_id_filter: str | None = None) -> None:
    """Render the memory browser/list.

    Args:
        memory_client: The Memory client instance
        user_id_filter: Optional user ID to filter memories
    """
    st.header("Browse Memories")

    # Pagination state
    if "page" not in st.session_state:
        st.session_state.page = 0

    page_size = 10

    # Bulk actions
    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    try:
        # Get memories
        memories = memory_client.get_all(
            user_id=user_id_filter,
            limit=100,
        )

        if not memories:
            st.info("No memories found." + (" Try removing the user filter." if user_id_filter else ""))
            return

        # Pagination
        total_pages = (len(memories) + page_size - 1) // page_size
        start_idx = st.session_state.page * page_size
        end_idx = min(start_idx + page_size, len(memories))

        st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(memories)} memories")

        # Bulk delete
        if "selected_for_delete" not in st.session_state:
            st.session_state.selected_for_delete = set()

        # Display memories
        for memory in memories[start_idx:end_idx]:
            memory_id = memory.get("id", "")
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})

            render_memory_item(memory_client, memory_id, content, metadata)

        # Pagination controls
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("Previous", disabled=st.session_state.page == 0):
                st.session_state.page -= 1
                st.rerun()

        with col2:
            st.caption(f"Page {st.session_state.page + 1} of {total_pages}")

        with col3:
            if st.button("Next", disabled=st.session_state.page >= total_pages - 1):
                st.session_state.page += 1
                st.rerun()

    except Exception as e:
        st.error(f"Failed to load memories: {e}")


def render_memory_item(
    memory_client: Memory,
    memory_id: str,
    content: str,
    metadata: dict,
) -> None:
    """Render a single memory item in the list.

    Args:
        memory_client: The Memory client instance
        memory_id: The memory ID
        content: The memory content
        metadata: The memory metadata
    """
    content_type = metadata.get("content_type", "")
    image_path = metadata.get("image_path", "")
    file_path = metadata.get("file_path", "")

    with st.container(border=True):
        # Header row
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            user_id = metadata.get("user_id", "")
            agent_id = metadata.get("agent_id", "")
            created_at = metadata.get("created_at", "")

            badges = []
            if content_type:
                type_display = content_type.upper() if content_type == "pdf" else content_type
                badges.append(f"Type: {type_display}")
            if user_id:
                badges.append(f"User: {user_id}")
            if agent_id:
                badges.append(f"Agent: {agent_id}")
            if created_at:
                # Format date nicely
                date_str = created_at[:10] if len(created_at) >= 10 else created_at
                badges.append(f"Created: {date_str}")

            if badges:
                st.caption(" | ".join(badges))

        with col2:
            view_key = f"view_{memory_id}"
            if st.button("View", key=view_key, use_container_width=True):
                st.session_state[f"expanded_{memory_id}"] = True

        with col3:
            delete_key = f"del_list_{memory_id}"
            if st.button("Delete", key=delete_key, type="secondary", use_container_width=True):
                try:
                    # Delete the image file if it exists
                    if content_type == "image" and image_path and Path(image_path).exists():
                        Path(image_path).unlink()

                    memory_client.delete(memory_id)
                    st.toast("Memory deleted!", icon="🗑️")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

        # Show thumbnail for images or PDF info
        if content_type == "image" and image_path and Path(image_path).exists():
            st.image(image_path, width=150)
        elif content_type == "pdf":
            page_count = metadata.get("page_count", "?")
            chunk_index = metadata.get("chunk_index", 0)
            total_chunks = metadata.get("total_chunks", 1)
            st.caption(f"PDF: {page_count} pages | Chunk {chunk_index + 1}/{total_chunks}")

        # Content preview
        preview = content[:150] + "..." if len(content) > 150 else content
        st.write(preview)

        # Expanded view
        if st.session_state.get(f"expanded_{memory_id}", False):
            st.divider()
            render_memory_card(memory_client, memory_id, content, metadata)
            if st.button("Close", key=f"close_{memory_id}"):
                st.session_state[f"expanded_{memory_id}"] = False
                st.rerun()
