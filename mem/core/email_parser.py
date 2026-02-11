"""Email parsing for EML and MSG formats."""

import email
import hashlib
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import List, Optional

import extract_msg

from mem.models.schemas import Attachment, EmailDocument


def _parse_address_list(value: Optional[str]) -> List[str]:
    """Parse comma-separated email addresses."""
    if not value:
        return []
    addresses = []
    for addr in value.split(","):
        _, email_addr = parseaddr(addr.strip())
        if email_addr:
            addresses.append(email_addr)
    return addresses


def _extract_text_body(msg: EmailMessage) -> Optional[str]:
    """Extract plain text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        return payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        return payload.decode("utf-8", errors="replace")
        return None
    else:
        if msg.get_content_type() == "text/plain":
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    return payload.decode("utf-8", errors="replace")
        return None


def _extract_html_body(msg: EmailMessage) -> Optional[str]:
    """Extract HTML body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        return payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        return payload.decode("utf-8", errors="replace")
        return None
    else:
        if msg.get_content_type() == "text/html":
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    return payload.decode("utf-8", errors="replace")
        return None


def _extract_attachments(msg: EmailMessage) -> List[Attachment]:
    """Extract attachment metadata from email message."""
    attachments = []
    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition:
                filename = part.get_filename() or "unnamed"
                content_type = part.get_content_type()
                payload = part.get_payload(decode=True)
                size = len(payload) if payload else 0
                attachments.append(
                    Attachment(filename=filename, content_type=content_type, size=size)
                )
    return attachments


def _get_thread_id(msg: EmailMessage) -> Optional[str]:
    """Extract thread ID from References or In-Reply-To headers."""
    references = msg.get("References")
    if references:
        refs = references.split()
        if refs:
            return refs[0].strip("<>")

    in_reply_to = msg.get("In-Reply-To")
    if in_reply_to:
        return in_reply_to.strip().strip("<>")

    return None


def _generate_message_id(content: bytes) -> str:
    """Generate a message ID from content hash if none exists."""
    return hashlib.sha256(content).hexdigest()[:32]


def parse_eml(file_bytes: bytes) -> EmailDocument:
    """Parse an EML file and return EmailDocument."""
    msg = email.message_from_bytes(file_bytes, policy=policy.default)

    message_id = msg.get("Message-ID", "").strip("<>")
    if not message_id:
        message_id = _generate_message_id(file_bytes)

    date = None
    date_str = msg.get("Date")
    if date_str:
        try:
            date = parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

    _, sender_email = parseaddr(msg.get("From", ""))

    return EmailDocument(
        message_id=message_id,
        subject=msg.get("Subject"),
        body_text=_extract_text_body(msg),
        body_html=_extract_html_body(msg),
        sender=sender_email or None,
        to=_parse_address_list(msg.get("To")),
        cc=_parse_address_list(msg.get("Cc")),
        bcc=_parse_address_list(msg.get("Bcc")),
        date=date,
        thread_id=_get_thread_id(msg),
        attachments=_extract_attachments(msg),
    )


def parse_msg(file_bytes: bytes) -> EmailDocument:
    """Parse an MSG file and return EmailDocument."""
    msg = extract_msg.Message(file_bytes)

    message_id = msg.messageId or _generate_message_id(file_bytes)
    message_id = message_id.strip("<>")

    date = None
    if msg.date:
        if isinstance(msg.date, datetime):
            date = msg.date
        elif isinstance(msg.date, str):
            try:
                date = parsedate_to_datetime(msg.date)
            except (ValueError, TypeError):
                pass

    to_addresses: List[str] = []
    if msg.to:
        to_addresses = _parse_address_list(msg.to)

    cc_addresses: List[str] = []
    if msg.cc:
        cc_addresses = _parse_address_list(msg.cc)

    attachments = []
    for att in msg.attachments or []:
        if hasattr(att, "filename") and hasattr(att, "data"):
            attachments.append(
                Attachment(
                    filename=att.filename or "unnamed",
                    content_type=getattr(att, "mimetype", None),
                    size=len(att.data) if att.data else 0,
                )
            )

    return EmailDocument(
        message_id=message_id,
        subject=msg.subject,
        body_text=msg.body,
        body_html=msg.htmlBody,
        sender=msg.sender,
        to=to_addresses,
        cc=cc_addresses,
        bcc=[],
        date=date,
        thread_id=None,
        attachments=attachments,
    )


def parse_email(file_bytes: bytes, filename: str) -> EmailDocument:
    """Parse email file based on extension."""
    filename_lower = filename.lower()
    if filename_lower.endswith(".msg"):
        return parse_msg(file_bytes)
    elif filename_lower.endswith(".eml"):
        return parse_eml(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
