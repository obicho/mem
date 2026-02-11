"""Email CRUD endpoints."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from mem.client import Memory
from mem.core.chunker import ContentType, chunk_content
from mem.core.email_parser import parse_email
from mem.core.embeddings import embed_batch
from mem.db.chromadb import ChromaDBClient
from mem.dependencies import get_db, get_memory_client, verify_api_key
from mem.models.schemas import (
    APIResponse,
    EmailChunk,
    EmailListResponse,
    InboundEmailPayload,
    IngestResponse,
)

router = APIRouter(prefix="/emails", tags=["emails"])


@router.post(
    "/inbound",
    response_model=APIResponse,
    summary="Ingest email from JSON payload",
    description="Process a normalized email JSON payload and add to memory",
)
async def ingest_inbound_email(
    payload: InboundEmailPayload,
    _api_key: str = Depends(verify_api_key),
    memory: Memory = Depends(get_memory_client),
) -> APIResponse:
    """Ingest an email from a normalized JSON payload into memory."""
    if not payload.body or not payload.body.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email body cannot be empty",
        )

    # Generate message_id if not provided
    message_id = payload.message_id or f"inbound-{uuid.uuid4().hex[:12]}@memory"

    # Build email metadata (ChromaDB only accepts scalar values)
    email_metadata = {
        "message_id": message_id,
        "subject": payload.subject,
        "sender": payload.sender,
        "recipients": ", ".join(payload.recipients) if payload.recipients else "",
        "cc": ", ".join(payload.cc) if payload.cc else "",
    }
    if payload.date:
        email_metadata["date"] = payload.date.isoformat()
    if payload.thread_id:
        email_metadata["thread_id"] = payload.thread_id

    # Prepend subject for better context
    content = payload.body
    if payload.subject:
        content = f"Subject: {payload.subject}\n\n{payload.body}"

    try:
        result = memory.add(
            content=content,
            user_id=payload.user_id,
            agent_id=payload.agent_id,
            metadata=email_metadata,
            content_type="email",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store email: {str(e)}",
        )

    return APIResponse(
        success=True,
        data={
            "email_id": message_id,
            "memory_id": result.get("memory_id") or result.get("memory_ids", [None])[0],
            "chunks_created": result.get("chunks_created", 1),
            "status": result.get("status", "added"),
        },
    )


@router.post(
    "",
    response_model=APIResponse,
    summary="Ingest email file",
    description="Upload and process an EML or MSG email file",
)
async def ingest_email(
    file: UploadFile = File(..., description="Email file (EML or MSG format)"),
    _api_key: str = Depends(verify_api_key),
    db: ChromaDBClient = Depends(get_db),
) -> APIResponse:
    """Ingest an email file into the system."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".eml") or filename_lower.endswith(".msg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be EML or MSG format",
        )

    try:
        content = await file.read()
        email_doc = parse_email(content, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse email: {str(e)}",
        )

    # Build metadata from email document
    email_metadata = {
        "message_id": email_doc.message_id,
        "subject": email_doc.subject,
        "sender": email_doc.sender,
        "recipients": email_doc.recipients,
    }
    chunk_dicts = chunk_content(email_doc.body, email_metadata, ContentType.EMAIL)

    # Convert to EmailChunk objects for db.add_email
    chunks = [
        EmailChunk(
            chunk_id=f"{email_doc.message_id}_{i}",
            email_id=email_doc.message_id,
            content=chunk["content"],
            chunk_index=i,
            metadata=chunk.get("metadata", {}),
        )
        for i, chunk in enumerate(chunk_dicts)
    ]

    if chunks:
        try:
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embed_batch(chunk_texts)
            db.add_email(email_doc, chunks, embeddings)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store email: {str(e)}",
            )

    return APIResponse(
        success=True,
        data=IngestResponse(
            email_id=email_doc.message_id,
            chunks_created=len(chunks),
            status="ingested",
        ),
    )


@router.get(
    "",
    response_model=APIResponse,
    summary="List emails",
    description="Get a paginated list of ingested emails",
)
async def list_emails(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sender: Optional[str] = Query(None, description="Filter by sender email"),
    date_from: Optional[str] = Query(None, description="Filter by date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter by date (ISO format)"),
    _api_key: str = Depends(verify_api_key),
    db: ChromaDBClient = Depends(get_db),
) -> APIResponse:
    """List all ingested emails with pagination."""
    filters = {}
    if sender:
        filters["sender"] = sender
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    emails, total = db.list_emails(limit=limit, offset=offset, filters=filters or None)

    return APIResponse(
        success=True,
        data=EmailListResponse(
            emails=emails,
            total=total,
            limit=limit,
            offset=offset,
        ),
    )


@router.get(
    "/{email_id}",
    response_model=APIResponse,
    summary="Get email by ID",
    description="Retrieve a specific email with all its metadata",
)
async def get_email(
    email_id: str,
    _api_key: str = Depends(verify_api_key),
    db: ChromaDBClient = Depends(get_db),
) -> APIResponse:
    """Get an email by its ID."""
    email_data = db.get_email(email_id)

    if not email_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email not found: {email_id}",
        )

    return APIResponse(
        success=True,
        data=email_data,
    )


@router.delete(
    "/{email_id}",
    response_model=APIResponse,
    summary="Delete email",
    description="Remove an email from the system",
)
async def delete_email(
    email_id: str,
    _api_key: str = Depends(verify_api_key),
    db: ChromaDBClient = Depends(get_db),
) -> APIResponse:
    """Delete an email by its ID."""
    deleted = db.delete_email(email_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email not found: {email_id}",
        )

    return APIResponse(
        success=True,
        data={"email_id": email_id, "status": "deleted"},
    )
