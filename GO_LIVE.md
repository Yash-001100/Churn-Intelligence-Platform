# Make Your Dashboard Live (Shareable Link)

Follow these steps to get a **public link** so anyone can open your Churn Intelligence dashboard.

---

## 1. Open Streamlit Community Cloud

Go to: **https://share.streamlit.io**

---

## 2. Sign in with GitHub

- Click **Sign in with GitHub**
- Authorize Streamlit to access your GitHub account (if prompted)

---

## 3. Create a new app

- Click **New app**
- Fill in exactly:

| Field | Value |
|-------|--------|
| **Repository** | `Yash-001100/Churn-Intelligence-Platform` |
| **Branch** | `main` |
| **Main file path** | `app.py` |

- Click **Deploy**

---

## 4. Wait for the build

- The first build can take **2–5 minutes**
- If it fails, open the **Logs** and check for missing dependencies or path errors
- When you see **"Your app is live!"**, the dashboard is ready

---

## 5. Copy your shareable link

- At the top you’ll see your **App URL**, for example:  
  `https://churn-intelligence-platform-xxxxx.streamlit.app`
- **Copy this URL** — this is the link you share with others

---

## 6. Share the link

- Send the URL to anyone (e.g. email, chat, portfolio)
- They open it in a browser, see the **login screen**, and can **Sign up** or **Log in** to use the dashboard
- No installation needed on their side

---

## Optional: Email verification or Salesforce

If you use **email verification** (signup confirmation) or **Salesforce** in the app:

1. In share.streamlit.io, open your app
2. Click the **⋮** (three dots) next to the app
3. Go to **Settings** → **Secrets**
4. Add the same variables you have in `.env`, for example:

```toml
APP_URL = "https://your-actual-app-url.streamlit.app"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USER = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"
FROM_EMAIL = "your_email@gmail.com"
```

Use your **real App URL** from step 5 for `APP_URL`. Save; the app will use these as environment variables.

---

**That’s it.** Once deployed, your dashboard is live and anyone with the link can access it.
