"""
Dashboard authentication: SQLite-backed user store with hashed passwords.
Email verification: new users get a verification token until they click the link.
"""
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager

from .config import AUTH_DB_PATH, EMAIL_VERIFICATION_ENABLED

# Use passlib for secure password hashing; fallback to hashlib if not installed
try:
    from passlib.hash import pbkdf2_sha256
    def _hash_password(password: str) -> str:
        return pbkdf2_sha256.hash(password)
    def _verify_password(password: str, hash_str: str) -> bool:
        return pbkdf2_sha256.verify(password, hash_str)
except ImportError:
    import hashlib
    import secrets
    def _hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        h = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}${h}"
    def _verify_password(password: str, hash_str: str) -> bool:
        parts = hash_str.split("$", 1)
        if len(parts) != 2:
            return False
        salt, stored = parts[0], parts[1]
        h = hashlib.sha256((salt + password).encode()).hexdigest()
        return secrets.compare_digest(h, stored)


@contextmanager
def _get_conn():
    os.makedirs(os.path.dirname(AUTH_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create users table if it does not exist; add email column if missing."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN verification_token TEXT")
        except sqlite3.OperationalError:
            pass


def _validate_password(password: str) -> tuple[bool, str]:
    """
    Enforce signup password policy:
    - Minimum 8 characters (12+ recommended).
    - At least one uppercase (A–Z), lowercase (a–z), number (0–9), and symbol.
    Returns (valid, error_message).
    """
    if not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters (12+ recommended for better security)."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter (A–Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter (a–z)."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one number (0–9)."
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;':\",./<>?`~\\]", password):
        return False, "Password must include at least one symbol (e.g. !@#$)."
    return True, ""


def add_user(
    username: str, password: str, email: str = "", confirm_password: str = ""
) -> tuple[bool, str, str | None]:
    """
    Register a new user. Returns (success, message, verification_token or None).
    When email verification is enabled and email is provided, returns a token to send.
    Username is case-insensitive for lookup but stored as given.
    """
    username = (username or "").strip()
    email = (email or "").strip()
    if not username:
        return False, "Username is required.", None
    if len(username) < 2:
        return False, "Username must be at least 2 characters.", None
    valid, msg = _validate_password(password)
    if not valid:
        return False, msg, None
    if confirm_password is not None and password != confirm_password:
        return False, "Password and Confirm password do not match.", None
    if EMAIL_VERIFICATION_ENABLED and not email:
        return False, "Email is required for signup. We will send you a verification link.", None
    if email and "@" not in email:
        return False, "Please enter a valid email address.", None
    init_db()
    token = secrets.token_urlsafe(32) if (EMAIL_VERIFICATION_ENABLED and email) else None
    email_verified = 0 if token else 1
    try:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO users (username, email, password_hash, email_verified, verification_token)
                   VALUES (?, ?, ?, ?, ?)""",
                (username, email or None, _hash_password(password), email_verified, token),
            )
        if token:
            return True, "Account created. Please check your email for a verification link. Once verified, you can log in.", token
        return True, "Account created. You can log in now.", None
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please log in or choose another.", None


def verify_user(username: str, password: str) -> tuple[bool, str]:
    """
    Verify credentials. Returns (success, message).
    Lookup is case-insensitive. Login is blocked until email is verified (if verification is enabled).
    """
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            """SELECT username, password_hash, COALESCE(email_verified, 1) AS email_verified
               FROM users WHERE LOWER(username) = LOWER(?)""",
            (username,),
        ).fetchone()
    if not row:
        return False, "Username or password is incorrect."
    if not _verify_password(password, row["password_hash"]):
        return False, "Username or password is incorrect."
    if EMAIL_VERIFICATION_ENABLED and row["email_verified"] != 1:
        return False, "Please verify your email first. Check your inbox for the verification link, then log in."
    return True, row["username"]


def verify_email_token(token: str) -> tuple[bool, str]:
    """
    Mark the user as verified when they click the link. Returns (success, message).
    """
    if not token or not token.strip():
        return False, "Invalid verification link."
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE verification_token = ?",
            (token.strip(),),
        ).fetchone()
        if not row:
            return False, "Invalid or expired verification link."
        conn.execute(
            "UPDATE users SET email_verified = 1, verification_token = NULL WHERE verification_token = ?",
            (token.strip(),),
        )
    return True, "Email verified. You can log in now."


def delete_user(username: str, password: str) -> tuple[bool, str]:
    """
    Permanently delete a user account. Verifies credentials first.
    Removes username, email and password from the database.
    Returns (success, message).
    """
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT username, password_hash FROM users WHERE LOWER(username) = LOWER(?)",
            (username,),
        ).fetchone()
        if not row:
            return False, "Username or password is incorrect."
        if not _verify_password(password, row["password_hash"]):
            return False, "Username or password is incorrect."
        conn.execute("DELETE FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    return True, "Your account has been permanently deleted. Your username, email and password have been removed."
