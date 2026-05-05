# How to upload this project to GitHub

This guide walks you through publishing your project to GitHub for the first time.

## 🎯 What's in this folder

```
pdf-rag-chat/
├── README.md              ← project landing page (renders on GitHub)
├── LICENSE                ← MIT license
├── CONTRIBUTING.md        ← contributor guide
├── .gitignore             ← keeps secrets and junk out of git
├── .env.example           ← config template (no secrets)
├── requirements.txt       ← Python dependencies
│
├── app.py                 ← FastAPI server with built-in UI
├── ingest.py              ← CLI for batch indexing
│
├── rag/                   ← core RAG package
│   ├── __init__.py
│   ├── document_processor.py
│   ├── vector_store.py
│   └── pipeline.py
│
└── docs/                  ← extended documentation
    ├── ARCHITECTURE.md
    ├── CONFIGURATION.md
    ├── API.md
    └── TROUBLESHOOTING.md
```

---

## ✅ Before you upload — checklist

1. **Update the LICENSE** — open `LICENSE` and replace `[Your Name]` with your actual name
2. **Update README.md** — replace `<your-username>` in the clone URL with your GitHub username
3. **Add screenshots** (optional but highly recommended):
   - Run the app, take screenshots of the welcome screen and a chat session
   - Save them as `docs/screenshot-welcome.png` and `docs/screenshot-chat.png`
4. **Make sure no secrets are in your code** — no hardcoded API keys, passwords, etc.
5. **Test it works locally** one more time:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

---

## 📤 Step-by-step: upload to GitHub

### Option A — Using the GitHub website (easiest)

1. Go to [github.com](https://github.com) and click the **+** icon → **New repository**
2. Fill in:
   - **Repository name**: `pdf-rag-chat` (or whatever you like)
   - **Description**: "Local PDF RAG chat with Ollama, FastAPI, and FAISS"
   - **Public** or **Private** — your choice
   - ⚠️ **Don't** tick "Add README", "Add .gitignore", or "Add license" — we already have those
3. Click **Create repository**
4. On the next page, copy the URL shown (looks like `https://github.com/yourname/pdf-rag-chat.git`)
5. In your terminal, navigate to this folder:
   ```bash
   cd /path/to/pdf-rag-chat
   ```
6. Initialise git and push:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourname/pdf-rag-chat.git
   git push -u origin main
   ```

### Option B — Using GitHub Desktop (no terminal)

1. Download [GitHub Desktop](https://desktop.github.com)
2. **File** → **Add Local Repository** → select this folder
3. Click **Publish repository** at the top
4. Choose name, description, public/private, then **Publish**

### Option C — Using VS Code

1. Open this folder in VS Code
2. Click the **Source Control** icon in the left sidebar (looks like a branching tree)
3. Click **Initialize Repository**
4. Stage all files (click the `+` next to "Changes")
5. Type a commit message and click the **✓** to commit
6. Click **Publish to GitHub** at the top

---

## 🎨 Make it shine on GitHub

After uploading:

### 1. Add topics (tags) to make it discoverable
On your repo page, click the ⚙️ next to "About" on the right side, then add topics like:
- `rag`
- `langchain`
- `fastapi`
- `ollama`
- `local-llm`
- `faiss`
- `pdf-chat`
- `python`

### 2. Add a description and website
In the same "About" panel:
- Description: "Chat with your PDFs locally — RAG-powered, no API keys"
- Website: leave blank or add your demo URL if you have one

### 3. Enable Issues and Discussions
Settings → Features → tick **Issues** and **Discussions**

### 4. Pin it to your profile
Go to your GitHub profile → click "Customize your pins" → select this repo

---

## 🚀 After it's live

- **Share the URL** with others
- **Watch for issues** people open and respond when you can
- **Add a "Star" button shoutout** in your README's footer (already included)
- **Tag a release** when you're happy with a version:
  ```bash
  git tag v1.0.0
  git push --tags
  ```
  Then on GitHub: Releases → Draft a new release

---

## 🆘 Common upload issues

| Problem | Fix |
|---|---|
| `git: command not found` | Install Git from [git-scm.com](https://git-scm.com) |
| `fatal: remote origin already exists` | Run `git remote remove origin` then re-add |
| `Permission denied (publickey)` | Use HTTPS URL instead of SSH, or [set up SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) |
| Big files rejected | Make sure `.gitignore` excludes `vector_store/` and `*.pdf` |
| `git push` asks for credentials repeatedly | Set up a [Personal Access Token](https://github.com/settings/tokens) and use it as the password |

---

That's it! Happy publishing 🎉
