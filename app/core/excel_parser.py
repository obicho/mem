"""Excel parsing service for extracting data from Excel/CSV files."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ParsedSheet:
    """Parsed Excel sheet."""

    name: str
    headers: list[str]
    rows: list[list[str]]
    row_count: int
    markdown: str


@dataclass
class ParsedExcel:
    """Parsed Excel document."""

    sheets: list[ParsedSheet]
    sheet_count: int
    file_hash: str


class ExcelParser:
    """Service for parsing Excel/CSV files and converting to markdown tables."""

    def parse(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
    ) -> ParsedExcel:
        """Parse an Excel or CSV file and convert each sheet to markdown.

        Args:
            file_path: Path to the Excel/CSV file.
            file_bytes: Raw file bytes (alternative to file_path).

        Returns:
            ParsedExcel with sheets converted to markdown tables.

        Raises:
            ValueError: If no input is provided or file format is unsupported.
        """
        if not file_path and not file_bytes:
            raise ValueError("Must provide either file_path or file_bytes")

        # Get bytes for hashing
        if file_path:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
            file_ext = Path(file_path).suffix.lower()
        else:
            raw_bytes = file_bytes
            file_ext = None

        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:32]

        # Determine file type and parse
        sheets = []

        if file_ext == ".csv" or (file_bytes and self._is_csv(file_bytes)):
            # CSV file - single sheet
            if file_path:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))

            parsed_sheet = self._dataframe_to_sheet(df, "Sheet1")
            if parsed_sheet:
                sheets.append(parsed_sheet)
        else:
            # Excel file - multiple sheets possible
            if file_path:
                excel_file = pd.ExcelFile(file_path)
            else:
                excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                parsed_sheet = self._dataframe_to_sheet(df, sheet_name)
                if parsed_sheet:
                    sheets.append(parsed_sheet)

        return ParsedExcel(
            sheets=sheets,
            sheet_count=len(sheets),
            file_hash=file_hash,
        )

    def _is_csv(self, data: bytes) -> bool:
        """Check if bytes look like CSV data."""
        try:
            # Try to decode as text and check for CSV patterns
            text = data[:1000].decode("utf-8", errors="ignore")
            lines = text.strip().split("\n")
            if len(lines) < 2:
                return False
            # Check if lines have consistent comma counts
            comma_counts = [line.count(",") for line in lines[:5]]
            return len(set(comma_counts)) == 1 and comma_counts[0] > 0
        except Exception:
            return False

    def _dataframe_to_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str,
    ) -> Optional[ParsedSheet]:
        """Convert a pandas DataFrame to a ParsedSheet.

        Args:
            df: The DataFrame to convert.
            sheet_name: Name of the sheet.

        Returns:
            ParsedSheet or None if the sheet is empty.
        """
        # Drop completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            return None

        # Convert headers to strings
        headers = [str(col) for col in df.columns]

        # Convert all values to strings
        rows = []
        for _, row in df.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append("")
                else:
                    row_values.append(str(val))
            rows.append(row_values)

        if not rows:
            return None

        # Convert to markdown
        markdown = self._sheet_to_markdown(headers, rows)

        return ParsedSheet(
            name=sheet_name,
            headers=headers,
            rows=rows,
            row_count=len(rows),
            markdown=markdown,
        )

    def _sheet_to_markdown(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> str:
        """Convert headers and rows to a markdown table string.

        Args:
            headers: Column headers.
            rows: Row data.

        Returns:
            Markdown table string.
        """
        lines = []

        # Header row
        header_line = "| " + " | ".join(headers) + " |"
        lines.append(header_line)

        # Separator row
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines.append(separator)

        # Data rows
        for row in rows:
            # Ensure row has same number of columns as headers
            padded_row = row + [""] * (len(headers) - len(row))
            # Escape pipe characters in cell values
            escaped_row = [cell.replace("|", "\\|") for cell in padded_row[:len(headers)]]
            row_line = "| " + " | ".join(escaped_row) + " |"
            lines.append(row_line)

        return "\n".join(lines)
