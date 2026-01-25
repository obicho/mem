"""Tests for email parser."""

import pytest

from app.core.email_parser import parse_eml
from app.models.schemas import EmailDocument


SAMPLE_EML = b"""From: sender@example.com
To: recipient@example.com
Cc: cc@example.com
Subject: Test Email Subject
Date: Mon, 15 Jan 2024 10:30:00 +0000
Message-ID: <test123@example.com>
References: <thread001@example.com>
Content-Type: text/plain; charset="utf-8"

This is the body of the test email.
It contains multiple lines.

Best regards,
Sender
"""


SAMPLE_MULTIPART_EML = b"""From: sender@example.com
To: recipient@example.com
Subject: Multipart Test
Date: Mon, 15 Jan 2024 10:30:00 +0000
Message-ID: <multipart123@example.com>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset="utf-8"

Plain text version of the email.

--boundary123
Content-Type: text/html; charset="utf-8"

<html><body><p>HTML version of the email.</p></body></html>

--boundary123--
"""


def test_parse_eml_basic():
    """Test basic EML parsing."""
    result = parse_eml(SAMPLE_EML)

    assert isinstance(result, EmailDocument)
    assert result.message_id == "test123@example.com"
    assert result.subject == "Test Email Subject"
    assert result.sender == "sender@example.com"
    assert "recipient@example.com" in result.to
    assert "cc@example.com" in result.cc
    assert "This is the body" in result.body_text
    assert result.thread_id == "thread001@example.com"


def test_parse_eml_multipart():
    """Test multipart EML parsing."""
    result = parse_eml(SAMPLE_MULTIPART_EML)

    assert result.message_id == "multipart123@example.com"
    assert result.body_text is not None
    assert "Plain text version" in result.body_text
    assert result.body_html is not None
    assert "HTML version" in result.body_html


def test_parse_eml_generates_message_id():
    """Test that message ID is generated when missing."""
    eml_without_id = b"""From: sender@example.com
To: recipient@example.com
Subject: No Message ID
Content-Type: text/plain

Body text here.
"""
    result = parse_eml(eml_without_id)

    assert result.message_id is not None
    assert len(result.message_id) > 0


def test_parse_eml_handles_empty_fields():
    """Test handling of emails with missing fields."""
    minimal_eml = b"""From: sender@example.com
Content-Type: text/plain

Minimal email body.
"""
    result = parse_eml(minimal_eml)

    assert result.sender == "sender@example.com"
    assert result.to == []
    assert result.cc == []
    assert result.subject is None
