import json
import streamlit as st
from app.client import Memory


def render_memory_card(
    memory_client: Memory,
    memory_id: str,
    content: str,
    metadata: dict,
) -> None:
    """Render a single memory card with view/edit capabilities.

    Args:
        memory_client: The Memory client instance
        memory_id: The memory ID
        content: The memory content
        metadata: The memory metadata
    """
    edit_key = f"edit_mode_{memory_id}"

    # Check if we're in edit mode
    is_editing = st.session_state.get(edit_key, False)

    if is_editing:
        render_edit_mode(memory_client, memory_id, content, metadata, edit_key)
    else:
        render_view_mode(memory_client, memory_id, content, metadata, edit_key)


def render_view_mode(
    memory_client: Memory,
    memory_id: str,
    content: str,
    metadata: dict,
    edit_key: str,
) -> None:
    """Render memory in view mode.

    Args:
        memory_client: The Memory client instance
        memory_id: The memory ID
        content: The memory content
        metadata: The memory metadata
        edit_key: The session state key for edit mode
    """
    # Memory ID
    st.caption(f"ID: {memory_id}")

    # Full content
    st.subheader("Content")
    st.write(content)

    # Metadata
    st.subheader("Metadata")

    # Filter out internal metadata for display
    display_metadata = {k: v for k, v in metadata.items()}
    st.json(display_metadata)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Edit", key=f"edit_btn_{memory_id}", use_container_width=True):
            st.session_state[edit_key] = True
            st.rerun()

    with col2:
        if st.button("Delete", key=f"del_card_{memory_id}", type="secondary", use_container_width=True):
            if st.session_state.get(f"confirm_delete_{memory_id}", False):
                try:
                    memory_client.delete(memory_id)
                    st.toast("Memory deleted!", icon="")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
            else:
                st.session_state[f"confirm_delete_{memory_id}"] = True
                st.warning("Click Delete again to confirm")
                st.rerun()


def render_edit_mode(
    memory_client: Memory,
    memory_id: str,
    content: str,
    metadata: dict,
    edit_key: str,
) -> None:
    """Render memory in edit mode.

    Args:
        memory_client: The Memory client instance
        memory_id: The memory ID
        content: The memory content
        metadata: The memory metadata
        edit_key: The session state key for edit mode
    """
    st.caption(f"ID: {memory_id}")

    # Editable content
    new_content = st.text_area(
        "Content",
        value=content,
        height=200,
        key=f"content_edit_{memory_id}",
    )

    # Editable metadata (as JSON)
    # Filter out read-only metadata
    editable_metadata = {
        k: v for k, v in metadata.items()
        if k not in ["created_at", "updated_at", "user_id", "agent_id", "run_id"]
    }

    metadata_str = st.text_area(
        "Custom Metadata (JSON)",
        value=json.dumps(editable_metadata, indent=2) if editable_metadata else "{}",
        height=100,
        key=f"metadata_edit_{memory_id}",
        help="Edit custom metadata. System fields (created_at, user_id, etc.) are preserved.",
    )

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save", key=f"save_{memory_id}", type="primary", use_container_width=True):
            # Parse metadata
            try:
                new_metadata = json.loads(metadata_str) if metadata_str.strip() else {}
                if not isinstance(new_metadata, dict):
                    st.error("Metadata must be a JSON object")
                    return
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return

            # Update memory
            try:
                memory_client.update(
                    memory_id=memory_id,
                    content=new_content,
                    metadata=new_metadata if new_metadata else None,
                )
                st.session_state[edit_key] = False
                st.toast("Memory updated!", icon="")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update: {e}")

    with col2:
        if st.button("Cancel", key=f"cancel_{memory_id}", use_container_width=True):
            st.session_state[edit_key] = False
            st.rerun()
