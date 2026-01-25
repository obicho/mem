"""Email CRUD endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.core.chunker import chunk_email
from app.core.email_parser import parse_email
from app.core.embeddings import embed_batch
from app.db.chromadb import ChromaDBClient
from app.dependencies import get_db, verify_api_key
from app.models.schemas import APIResponse, EmailListResponse, IngestResponse

router = APIRouter(prefix="/emails", tags=["emails"])


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

    chunks = chunk_email(email_doc)

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
