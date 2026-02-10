"""Tests for Excel parser."""

import io
import pytest
import pandas as pd

from app.core.excel_parser import ExcelParser, ParsedExcel, ParsedSheet


@pytest.fixture
def excel_parser():
    """Create an ExcelParser instance."""
    return ExcelParser()


@pytest.fixture
def sample_xlsx_bytes():
    """Create sample Excel bytes with a single sheet."""
    df = pd.DataFrame({
        "Supplier": ["NSL Industry", "Asahi-Thai", "DALI Corp"],
        "Location": ["Thailand", "Thailand", "Japan"],
        "Contact": ["contact@nsl.com", "sales@asahi.co.th", "info@dali.jp"],
    })
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Suppliers")
    return buffer.getvalue()


@pytest.fixture
def multi_sheet_xlsx_bytes():
    """Create sample Excel bytes with multiple sheets."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df1 = pd.DataFrame({
            "Name": ["Alice", "Bob"],
            "Email": ["alice@example.com", "bob@example.com"],
        })
        df1.to_excel(writer, index=False, sheet_name="Contacts")

        df2 = pd.DataFrame({
            "Product": ["Widget A", "Widget B"],
            "Price": ["$10", "$20"],
        })
        df2.to_excel(writer, index=False, sheet_name="Products")

    return buffer.getvalue()


@pytest.fixture
def sample_csv_bytes():
    """Create sample CSV bytes."""
    csv_content = "Name,Company,Revenue\nAcme Inc,Technology,$1M\nBeta Corp,Finance,$500K\n"
    return csv_content.encode("utf-8")


def test_parse_xlsx_single_sheet(excel_parser, sample_xlsx_bytes):
    """Test parsing a single-sheet Excel file."""
    result = excel_parser.parse(file_bytes=sample_xlsx_bytes)

    assert isinstance(result, ParsedExcel)
    assert result.sheet_count == 1
    assert len(result.sheets) == 1
    assert result.file_hash is not None
    assert len(result.file_hash) == 32

    sheet = result.sheets[0]
    assert sheet.name == "Suppliers"
    assert sheet.row_count == 3
    assert "Supplier" in sheet.headers
    assert "Location" in sheet.headers


def test_parse_xlsx_multiple_sheets(excel_parser, multi_sheet_xlsx_bytes):
    """Test parsing a multi-sheet Excel file."""
    result = excel_parser.parse(file_bytes=multi_sheet_xlsx_bytes)

    assert result.sheet_count == 2
    assert len(result.sheets) == 2

    sheet_names = [s.name for s in result.sheets]
    assert "Contacts" in sheet_names
    assert "Products" in sheet_names

    # Find contacts sheet
    contacts = next(s for s in result.sheets if s.name == "Contacts")
    assert contacts.row_count == 2
    assert "Name" in contacts.headers
    assert "Email" in contacts.headers


def test_parse_csv(excel_parser, sample_csv_bytes):
    """Test parsing a CSV file."""
    result = excel_parser.parse(file_bytes=sample_csv_bytes)

    assert result.sheet_count == 1
    sheet = result.sheets[0]
    assert sheet.name == "Sheet1"  # Default name for CSV
    assert sheet.row_count == 2
    assert "Name" in sheet.headers
    assert "Company" in sheet.headers
    assert "Revenue" in sheet.headers


def test_sheet_to_markdown(excel_parser, sample_xlsx_bytes):
    """Test markdown conversion format."""
    result = excel_parser.parse(file_bytes=sample_xlsx_bytes)
    sheet = result.sheets[0]

    # Check markdown structure
    lines = sheet.markdown.split("\n")
    assert len(lines) >= 4  # Header + separator + data rows

    # Header row
    assert lines[0].startswith("|")
    assert "Supplier" in lines[0]
    assert "Location" in lines[0]

    # Separator row
    assert "---" in lines[1]

    # Data rows
    assert "NSL Industry" in sheet.markdown
    assert "Thailand" in sheet.markdown


def test_empty_sheet_skipped(excel_parser):
    """Test that empty sheets are skipped."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df1 = pd.DataFrame({"Name": ["Alice"]})
        df1.to_excel(writer, index=False, sheet_name="Data")

        df2 = pd.DataFrame()  # Empty
        df2.to_excel(writer, index=False, sheet_name="Empty")

    result = excel_parser.parse(file_bytes=buffer.getvalue())

    # Only the non-empty sheet should be included
    assert result.sheet_count == 1
    assert result.sheets[0].name == "Data"


def test_headers_extracted(excel_parser, sample_xlsx_bytes):
    """Test that headers are correctly extracted from first row."""
    result = excel_parser.parse(file_bytes=sample_xlsx_bytes)
    sheet = result.sheets[0]

    assert sheet.headers == ["Supplier", "Location", "Contact"]


def test_rows_extracted(excel_parser, sample_xlsx_bytes):
    """Test that rows are correctly extracted."""
    result = excel_parser.parse(file_bytes=sample_xlsx_bytes)
    sheet = result.sheets[0]

    assert len(sheet.rows) == 3
    assert sheet.rows[0][0] == "NSL Industry"
    assert sheet.rows[1][1] == "Thailand"
    assert sheet.rows[2][2] == "info@dali.jp"


def test_parse_no_input_raises_error(excel_parser):
    """Test that parsing without input raises an error."""
    with pytest.raises(ValueError, match="Must provide either file_path or file_bytes"):
        excel_parser.parse()


def test_file_hash_deterministic(excel_parser, sample_xlsx_bytes):
    """Test that file hash is deterministic."""
    result1 = excel_parser.parse(file_bytes=sample_xlsx_bytes)
    result2 = excel_parser.parse(file_bytes=sample_xlsx_bytes)

    assert result1.file_hash == result2.file_hash


def test_handles_na_values(excel_parser):
    """Test that NA/null values are handled."""
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", None],
        "Email": ["alice@test.com", None, "charlie@test.com"],
    })
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)

    result = excel_parser.parse(file_bytes=buffer.getvalue())
    sheet = result.sheets[0]

    # NA values should be empty strings
    assert sheet.rows[1][1] == ""  # Bob's missing email
    assert sheet.rows[2][0] == ""  # Missing name


def test_pipe_characters_escaped(excel_parser):
    """Test that pipe characters in cell values are escaped."""
    df = pd.DataFrame({
        "Command": ["cat file | grep foo", "ls -la"],
        "Description": ["Filter output", "List files"],
    })
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)

    result = excel_parser.parse(file_bytes=buffer.getvalue())
    markdown = result.sheets[0].markdown

    # Pipe in cell should be escaped
    assert "\\|" in markdown


def test_parse_xlsx_with_file_path(excel_parser, tmp_path, sample_xlsx_bytes):
    """Test parsing Excel file from path."""
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(sample_xlsx_bytes)

    result = excel_parser.parse(file_path=str(file_path))

    assert result.sheet_count == 1
    assert result.sheets[0].name == "Suppliers"


def test_parse_csv_with_file_path(excel_parser, tmp_path, sample_csv_bytes):
    """Test parsing CSV file from path."""
    file_path = tmp_path / "test.csv"
    file_path.write_bytes(sample_csv_bytes)

    result = excel_parser.parse(file_path=str(file_path))

    assert result.sheet_count == 1
    assert "Name" in result.sheets[0].headers
