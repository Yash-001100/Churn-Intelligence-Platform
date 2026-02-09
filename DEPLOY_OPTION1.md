# Deploy with Option 1: Streamlit Community Cloud

Follow these steps to put your dashboard live so anyone can open it via a link.

---

## Step 1: Create a GitHub repository

You already created the repo: **https://github.com/Yash-001100/Churn-Intelligence-Platform**

If you were starting from scratch: go to [github.com/new](https://github.com/new), set repository name (e.g. `Churn-Intelligence-Platform`), choose **Public**, do **not** initialize with a README, then **Create repository**.

---

## Step 2: Push your project to GitHub

In a terminal, from your **project folder** (where `app.py` and `requirements.txt` are):

```powershell
# Initialize git (if not already)
git init

# Add all files (respects .gitignore — .env and data/users.db are excluded)
git add .
git status

# Commit
git commit -m "Initial commit: Churn Intelligence Platform dashboard"

# Use main branch
git branch -M main

# Add your repo
git remote add origin https://github.com/Yash-001100/Churn-Intelligence-Platform.git

# Push
git push -u origin main
```

If GitHub asks for login, use a **Personal Access Token** (Settings → Developer settings → Personal access tokens) as the password, or sign in with GitHub CLI.

---

## Step 3: Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**.
2. Click **Sign in with GitHub** and authorize.
3. Click **New app**.
4. Fill in:
   - **Repository:** `Yash-001100/Churn-Intelligence-Platform`.
   - **Branch:** `main`.
   - **Main file path:** `app.py`.
5. Click **Deploy**.
6. Wait for the build (a few minutes). If it fails, check the logs for missing dependencies or path errors.
7. When it’s running, copy the **App URL** (e.g. `https://churn-intelligence-dashboard-main-xxxxx.streamlit.app`).

---

## Step 4: Share the link

- Share the **App URL** with anyone. They open it in a browser, see the login screen, and can sign up or log in to use the dashboard.
- You can paste this link in your portfolio, resume, or docs.

---

## Optional: Set secrets (email verification, Salesforce)

If you use email verification or Salesforce, set the same variables you have in `.env` in Streamlit:

1. In **[share.streamlit.io](https://share.streamlit.io)**, open your app.
2. Click the **⋮** menu → **Settings** → **Secrets**.
3. Add key-value pairs, for example:

```toml
APP_URL = "https://your-app-name.streamlit.app"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USER = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"
FROM_EMAIL = "your_email@gmail.com"
```

Save. The app will use these as environment variables (no `.env` file on the cloud).

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create a new repo on GitHub. |
| 2 | From project folder: `git init`, `git add .`, `git commit`, `git remote add origin ...`, `git push -u origin main`. |
| 3 | Go to share.streamlit.io → New app → select repo, branch `main`, file `app.py` → Deploy. |
| 4 | Copy the App URL and share it. |

Your dashboard is then live at that URL for anyone to use.
