import json
import streamlit as st
from app.client import Memory


def render_add_memory(memory_client: Memory) -> None:
    """Render the add memory form.

    Args:
        memory_client: The Memory client instance
    """
    st.header("Add Memory")

    with st.form("add_memory_form", clear_on_submit=True):
        # Content input
        content = st.text_area(
            "Memory Content",
            placeholder="Enter the memory content...",
            height=150,
            help="The text content of the memory",
        )

        # Optional fields in columns
        col1, col2 = st.columns(2)

        with col1:
            user_id = st.text_input(
                "User ID (optional)",
                placeholder="e.g., alice",
                help="Identifier for the user this memory belongs to",
            )

        with col2:
            agent_id = st.text_input(
                "Agent ID (optional)",
                placeholder="e.g., agent_123",
                help="Identifier for the agent that created this memory",
            )

        # Metadata JSON input
        metadata_str = st.text_area(
            "Custom Metadata (optional)",
            placeholder='{"key": "value"}',
            height=100,
            help="Additional metadata as JSON",
        )

        # Submit button
        submitted = st.form_submit_button("Add Memory", type="primary", use_container_width=True)

        if submitted:
            if not content.strip():
                st.error("Memory content cannot be empty")
                return

            # Parse metadata
            metadata = None
            if metadata_str.strip():
                try:
                    metadata = json.loads(metadata_str)
                    if not isinstance(metadata, dict):
                        st.error("Metadata must be a JSON object")
                        return
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON metadata: {e}")
                    return

            # Add memory
            try:
                result = memory_client.add(
                    content=content,
                    user_id=user_id if user_id else None,
                    agent_id=agent_id if agent_id else None,
                    metadata=metadata,
                )
                st.success(f"Memory added successfully! ID: {result['memory_id']}")
                st.toast("Memory added!", icon="")
            except Exception as e:
                st.error(f"Failed to add memory: {e}")
