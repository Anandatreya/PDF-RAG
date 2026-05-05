# Contributing to PDF RAG Chat

First off, thank you for taking the time to contribute! 🎉

## How can I contribute?

### 🐛 Reporting bugs

Before opening a bug report, please:
1. Check the [open issues](../../issues) to see if it's already reported
2. Try the steps in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
3. Make sure you're running the latest version

When opening a bug, please include:
- Your OS and Python version
- Your Ollama version (`ollama --version`)
- The exact command or action that triggered the bug
- The full error message and stack trace
- Steps to reproduce

### 💡 Suggesting features

Open a [discussion](../../discussions) first to gauge interest before opening a PR. Describe:
- What problem the feature solves
- How you imagine it working
- Any alternatives you've considered

### 🔧 Submitting code

1. **Fork** the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/my-amazing-feature
   ```

2. **Set up your dev environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt    # if it exists
   ```

3. **Make your changes** following the style guide below.

4. **Test your changes locally**:
   ```bash
   uvicorn app:app --reload
   # Try uploading a PDF and chatting
   ```

5. **Commit** with a clear message:
   ```bash
   git commit -m "Add: streaming response support"
   git commit -m "Fix: chat_history type error on empty memory"
   git commit -m "Docs: clarify embedding model choice"
   ```

6. **Push** and open a Pull Request.

## 🎨 Style guide

### Python
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all function signatures
- Keep functions under ~50 lines where possible
- Add docstrings to public classes and methods
- Use `logger.info(...)` instead of `print()`

### Commits
- Use the imperative mood: "Add feature" not "Added feature"
- Prefix with the type: `Add:`, `Fix:`, `Docs:`, `Refactor:`, `Test:`
- Keep the first line under 72 characters

### Documentation
- Update `README.md` if you change the user-facing behaviour
- Update relevant files in `docs/` for technical changes
- Add a row to the config table if you introduce a new env variable

## 🤝 Code of conduct

Be kind, be patient, and assume good faith. We're all here to learn and build something useful together.
