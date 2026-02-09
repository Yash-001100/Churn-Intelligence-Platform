"""
Send verification email with link. Uses SMTP (e.g. Gmail).
Configure via env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, APP_URL, FROM_EMAIL.
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .config import APP_URL, EMAIL_VERIFICATION_ENABLED, FROM_EMAIL, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER


def send_verification_email(to_email: str, username: str, token: str) -> tuple[bool, str]:
    """
    Send a verification link to the user's email.
    Returns (success, error_message).
    """
    if not EMAIL_VERIFICATION_ENABLED:
        return False, "Email sending is not configured."
    if not to_email or not token:
        return False, "Missing email or token."
    link = f"{APP_URL}?token={token}"
    subject = "Verify your Churn Prediction Dashboard account"
    body_plain = f"""Hello {username},

Please verify your account by clicking the link below:

{link}

If you did not sign up, you can ignore this email.

— Churn Prediction Dashboard
"""
    body_html = f"""<!DOCTYPE html>
<html>
<body>
<p>Hello <strong>{username}</strong>,</p>
<p>Please verify your account by clicking the link below:</p>
<p><a href="{link}">{link}</a></p>
<p>If you did not sign up, you can ignore this email.</p>
<p>— Churn Prediction Dashboard</p>
</body>
</html>
"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        msg.attach(MIMEText(body_plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)
