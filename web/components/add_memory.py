import json
import tempfile
from pathlib import Path

import streamlit as st
from app.client import Memory


def render_add_memory(memory_client: Memory) -> None:
    """Render the add memory form with support for text, images, and PDFs.

    Args:
        memory_client: The Memory client instance
    """
    st.header("Add Memory")

    # Tabs for different content types
    tab_text, tab_image, tab_pdf = st.tabs(["Text", "Image", "PDF"])

    with tab_text:
        render_text_form(memory_client)

    with tab_image:
        render_image_form(memory_client)

    with tab_pdf:
        render_pdf_form(memory_client)


def render_text_form(memory_client: Memory) -> None:
    """Render the text memory form."""
    with st.form("add_text_form", clear_on_submit=True):
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
                key="text_user_id",
            )

        with col2:
            agent_id = st.text_input(
                "Agent ID (optional)",
                placeholder="e.g., agent_123",
                help="Identifier for the agent that created this memory",
                key="text_agent_id",
            )

        # Metadata JSON input
        metadata_str = st.text_area(
            "Custom Metadata (optional)",
            placeholder='{"key": "value"}',
            height=100,
            help="Additional metadata as JSON",
            key="text_metadata",
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
                if "memory_ids" in result:
                    st.success(f"Memory added! Created {result['chunks_created']} chunks.")
                else:
                    st.success(f"Memory added! ID: {result['memory_id']}")
                st.toast("Memory added!", icon="✅")
            except Exception as e:
                st.error(f"Failed to add memory: {e}")


def render_image_form(memory_client: Memory) -> None:
    """Render the image upload form."""
    with st.form("add_image_form", clear_on_submit=True):
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            help="Upload an image to add to memory. A caption will be auto-generated.",
        )

        # Optional fields in columns
        col1, col2 = st.columns(2)

        with col1:
            user_id = st.text_input(
                "User ID (optional)",
                placeholder="e.g., alice",
                help="Identifier for the user this memory belongs to",
                key="image_user_id",
            )

        with col2:
            agent_id = st.text_input(
                "Agent ID (optional)",
                placeholder="e.g., agent_123",
                help="Identifier for the agent that created this memory",
                key="image_agent_id",
            )

        # Metadata JSON input
        metadata_str = st.text_area(
            "Custom Metadata (optional)",
            placeholder='{"key": "value"}',
            height=100,
            help="Additional metadata as JSON",
            key="image_metadata",
        )

        # Submit button
        submitted = st.form_submit_button(
            "Upload & Add Image",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if not uploaded_file:
                st.error("Please upload an image")
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

            # Add image to memory
            try:
                with st.spinner("Analyzing image and generating caption..."):
                    # Read image bytes
                    image_bytes = uploaded_file.read()

                    # Add custom metadata for original filename
                    if metadata is None:
                        metadata = {}
                    metadata["original_filename"] = uploaded_file.name

                    result = memory_client.add_image(
                        image_bytes=image_bytes,
                        user_id=user_id if user_id else None,
                        agent_id=agent_id if agent_id else None,
                        metadata=metadata,
                        filename=uploaded_file.name,
                    )

                st.success(f"Image added! ID: {result['memory_id']}")
                st.info(f"Generated caption: {result['caption'][:200]}...")
                st.toast("Image added!", icon="🖼️")
            except Exception as e:
                st.error(f"Failed to add image: {e}")


def render_pdf_form(memory_client: Memory) -> None:
    """Render the PDF upload form."""
    with st.form("add_pdf_form", clear_on_submit=True):
        # PDF upload
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload a PDF document. Text will be extracted and chunked for search.",
        )

        # Optional fields in columns
        col1, col2 = st.columns(2)

        with col1:
            user_id = st.text_input(
                "User ID (optional)",
                placeholder="e.g., alice",
                help="Identifier for the user this memory belongs to",
                key="pdf_user_id",
            )

        with col2:
            agent_id = st.text_input(
                "Agent ID (optional)",
                placeholder="e.g., agent_123",
                help="Identifier for the agent that created this memory",
                key="pdf_agent_id",
            )

        # Metadata JSON input
        metadata_str = st.text_area(
            "Custom Metadata (optional)",
            placeholder='{"key": "value"}',
            height=100,
            help="Additional metadata as JSON",
            key="pdf_metadata",
        )

        # Submit button
        submitted = st.form_submit_button(
            "Upload & Process PDF",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if not uploaded_file:
                st.error("Please upload a PDF file")
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

            # Add PDF to memory
            try:
                with st.spinner("Extracting text and processing PDF..."):
                    # Read PDF bytes
                    pdf_bytes = uploaded_file.read()

                    # Add custom metadata for original filename
                    if metadata is None:
                        metadata = {}
                    metadata["original_filename"] = uploaded_file.name

                    result = memory_client.add_pdf(
                        file_bytes=pdf_bytes,
                        user_id=user_id if user_id else None,
                        agent_id=agent_id if agent_id else None,
                        metadata=metadata,
                        filename=uploaded_file.name,
                    )

                st.success(
                    f"PDF added! {result['page_count']} pages, "
                    f"{result['chunks_created']} chunks created."
                )
                st.toast("PDF added!", icon="📄")
            except Exception as e:
                st.error(f"Failed to add PDF: {e}")
