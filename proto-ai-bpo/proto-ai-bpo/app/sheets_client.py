"""
sheets_client.py
=================
Thin, resilient wrapper around gspread that every other module uses to talk
to the three Google Sheets:

  1. Persona sheet   -> agent behavior / character / tone rules
  2. Company sheet    -> company details, products, policies, FAQs
  3. Ledger sheet     -> session / customer ledger (read + append)

Auth uses a Google Cloud service account (recommended for a server-side,
always-on BPO system — no browser OAuth dance, no token refresh babysitting).

Setup:
  1. Create a service account in Google Cloud Console, enable the
     "Google Sheets API" and "Google Drive API".
  2. Download the JSON key -> save as app/credentials/service_account.json
     (path configurable via GOOGLE_CREDENTIALS_PATH in .env).
  3. Share each of the 3 spreadsheets with the service account's email
     (found in the JSON key as "client_email") with Editor access.
"""

import logging
from functools import lru_cache
from typing import Any

import gspread
from google.oauth2.service_account import Credentials

from . import config

logger = logging.getLogger("proto_bpo.sheets_client")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


class SheetsClient:
    """Wraps a single gspread client shared across all sheet operations."""

    def __init__(self):
        self._gc = None

    def _client(self) -> gspread.Client:
        if self._gc is None:
            try:
                creds = Credentials.from_service_account_file(
                    config.GOOGLE_CREDENTIALS_PATH, scopes=SCOPES
                )
                self._gc = gspread.authorize(creds)
                logger.info("Google Sheets client authorized.")
            except FileNotFoundError:
                logger.error(
                    "Service account credentials not found at %s. "
                    "See sheets_client.py docstring for setup steps.",
                    config.GOOGLE_CREDENTIALS_PATH,
                )
                raise
        return self._gc

    def _open(self, sheet_key: str) -> gspread.Worksheet:
        cfg = config.SPREADSHEETS[sheet_key]
        sh = self._client().open_by_key(cfg["id"])
        return sh.worksheet(cfg["worksheet"])

    # -- generic helpers ----------------------------------------------------

    def get_all_records(self, sheet_key: str) -> list[dict[str, Any]]:
        """Return every row of a sheet as a list of {column: value} dicts."""
        try:
            ws = self._open(sheet_key)
            return ws.get_all_records()
        except Exception:
            logger.exception("Failed to read sheet '%s'", sheet_key)
            return []

    def append_row(self, sheet_key: str, row: list[Any]) -> bool:
        try:
            ws = self._open(sheet_key)
            ws.append_row(row, value_input_option="USER_ENTERED")
            return True
        except Exception:
            logger.exception("Failed to append row to sheet '%s'", sheet_key)
            return False

    def update_row_by_key(
        self, sheet_key: str, key_column: str, key_value: str, updates: dict[str, Any]
    ) -> bool:
        """Find the row where key_column == key_value and patch given columns."""
        try:
            ws = self._open(sheet_key)
            header = ws.row_values(1)
            records = ws.get_all_records()
            for idx, record in enumerate(records, start=2):  # row 1 = header
                if str(record.get(key_column)) == str(key_value):
                    for col_name, value in updates.items():
                        if col_name in header:
                            col_idx = header.index(col_name) + 1
                            ws.update_cell(idx, col_idx, value)
                    return True
            logger.warning(
                "Row with %s=%s not found in sheet '%s'; nothing updated.",
                key_column, key_value, sheet_key,
            )
            return False
        except Exception:
            logger.exception("Failed to update row in sheet '%s'", sheet_key)
            return False

    # -- domain-specific convenience wrappers --------------------------------

    def get_persona_rules(self) -> list[dict[str, Any]]:
        return self.get_all_records("persona")

    def get_company_details(self) -> list[dict[str, Any]]:
        return self.get_all_records("company")

    def get_ledger(self) -> list[dict[str, Any]]:
        return self.get_all_records("ledger")

    def append_ledger_entry(self, row: list[Any]) -> bool:
        return self.append_row("ledger", row)


@lru_cache(maxsize=1)
def get_sheets_client() -> SheetsClient:
    """Process-wide singleton so every module shares one authorized client."""
    return SheetsClient()
