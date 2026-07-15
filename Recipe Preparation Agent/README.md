# 🍳 Recipe Preparation Agent

> An AI-powered culinary assistant built with **Python Flask**, **IBM watsonx.ai**, and the **IBM Granite 3.3 8B Instruct** model. Uses a lightweight **Retrieval-Augmented Generation (RAG)** system to match your pantry ingredients against a recipe dataset and generate intelligent, waste-reducing cooking guidance.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Requirements](#-requirements)
4. [Installation](#-installation)
5. [Configuration](#-configuration)
6. [Running the App](#-running-the-app)
7. [Application Features](#-application-features)
8. [REST API Reference](#-rest-api-reference)
9. [RAG System Explained](#-rag-system-explained)
10. [Customising the Agent](#-customising-the-agent)
11. [Extending the Recipe Dataset](#-extending-the-recipe-dataset)
12. [Dietary Filter Tags](#-dietary-filter-tags)
13. [Production Deployment](#-production-deployment)
14. [Troubleshooting](#-troubleshooting)
15. [Security Notes](#-security-notes)

---

## 🧠 Project Overview

**Chef Aanya** is a multi-turn conversational recipe agent that:

- Accepts a list of ingredients you have at home (your **pantry**)
- Uses **TF-IDF cosine similarity** to retrieve the most relevant recipes from a built-in dataset (the RAG layer)
- Passes the top matches into a structured prompt sent to **IBM Granite 3.3 8B Instruct** via the `ibm-watsonx-ai` Python SDK
- Returns a full recipe with step-by-step instructions, smart substitutions for missing ingredients, and a **food-waste impact score (1–10)**
- Supports dietary filters (Vegetarian, Vegan, Gluten-Free, High-Protein, Quick) and allergy awareness
- Runs in **demo mode** (no IBM Cloud credentials required) for local testing

---

## 📁 Project Structure

```
recipe-preparation-agent/
│
├── app.py                  # Flask backend — RAG engine, watsonx.ai integration, REST API
├── requirements.txt        # All Python dependencies
├── .env.example            # Credential template (copy to .env and fill in)
├── .env                    # Your local secrets — NEVER commit this file
│
└── templates/
    └── index.html          # Single-page frontend — Bootstrap 5, dark mode, 4-tab UI
```

---

## 📦 Requirements

### System Requirements

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | **3.10+** | 3.11 or 3.12 recommended |
| pip | **23.0+** | Comes with Python |
| Internet connection | — | Required to reach IBM Cloud API |

### Python Dependencies

All dependencies are listed in [`requirements.txt`](requirements.txt):

| Package | Version | Purpose |
|---|---|---|
| `flask` | `>=3.0.0` | Web framework — routing, sessions, templating |
| `flask-session` | `>=0.6.0` | Server-side session support |
| `python-dotenv` | `>=1.0.0` | Load credentials from `.env` file |
| `ibm-watsonx-ai` | `>=1.0.0` | Official IBM SDK for watsonx.ai foundation models |
| `numpy` | `>=1.26.0` | Numerical arrays (required by scikit-learn) |
| `scikit-learn` | `>=1.4.0` | TF-IDF vectoriser and cosine similarity for RAG |
| `gunicorn` | `>=22.0.0` | Production WSGI server (optional for local dev) |

### IBM Cloud Requirements

| Requirement | Where to Get It |
|---|---|
| IBM Cloud account | [cloud.ibm.com](https://cloud.ibm.com/registration) — free tier available |
| IBM Cloud API Key | [cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys) |
| watsonx.ai project | [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → New Project |
| watsonx Project ID | Your project → **Manage** tab → **General** → copy the Project ID |

> **No IBM account?** Leave `IBM_CLOUD_API_KEY` blank in `.env`. The app runs in **demo mode** — recipe cards are shown from the local dataset without AI-generated text.

---

## ⚙️ Installation

Follow these steps exactly in order.

### Step 1 — Clone or download the project

```bash
# If using git:
git clone https://github.com/your-username/recipe-preparation-agent.git
cd recipe-preparation-agent

# Or simply navigate to the folder where the files were generated:
cd "Recipe Preparation Agent"
```

### Step 2 — Verify Python version

```bash
python --version
# Expected: Python 3.10.x or higher
```

**Python not found / `pip` not recognised?** This means Python is not installed or not on your PATH.

#### Automatic install (Windows — recommended)

Open **PowerShell** and run these two commands. They download and silently install Python 3.12.7 with pip and PATH registration built in:

```powershell
# 1. Download the official installer
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" `
  -OutFile "$env:TEMP\python-3.12.7-amd64.exe" -UseBasicParsing

# 2. Install silently — adds Python AND pip to PATH automatically
Start-Process -FilePath "$env:TEMP\python-3.12.7-amd64.exe" `
  -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0" `
  -Wait
```

After installation, **close and reopen PowerShell** (the PATH change only applies to new terminal windows), then run:

```powershell
python --version    # should print: Python 3.12.7
pip --version       # should print: pip 24.x from ...Python312\...
```

#### Manual install (Windows)
Download the installer from [python.org/downloads](https://www.python.org/downloads/), run it, and **tick both checkboxes**:
- ✅ Add Python to PATH
- ✅ Install pip

#### macOS / Linux
```bash
# macOS (via Homebrew)
brew install python@3.12

# Ubuntu / Debian
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip -y
```

### Step 3 — Create a virtual environment

A virtual environment keeps project dependencies isolated from your system Python.

```bash
# Create the environment
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate — macOS / Linux
source .venv/bin/activate
```

You should see `(.venv)` appear at the start of your terminal prompt.

### Step 4 — Install all dependencies

```bash
# Recommended — install everything at once
pip install -r requirements.txt
```

> **Windows note:** If `pip` still isn't found even after installing Python, use the module form — it always works:
> ```powershell
> python -m pip install -r requirements.txt
> ```

To install packages individually (for troubleshooting or reference):

```bash
# Core web framework
pip install "flask>=3.0.0"
pip install "flask-session>=0.6.0"

# Credential management
pip install "python-dotenv>=1.0.0"

# IBM watsonx.ai SDK  (installs ~40 sub-dependencies automatically)
pip install "ibm-watsonx-ai>=1.0.0"

# RAG / machine learning
pip install "numpy>=1.26.0"
pip install "scikit-learn>=1.4.0"

# Production server (optional — Linux/macOS only)
pip install "gunicorn>=22.0.0"
```

Verify the installation was successful:

```bash
python -c "
from importlib.metadata import version
for p in ['flask','flask-session','python-dotenv','ibm-watsonx-ai','numpy','scikit-learn']:
    print(p, version(p))
print('All packages OK')
"
```

---

## 🔐 Configuration

### Step 5 — Create your `.env` file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in any text editor and fill in your values:

```dotenv
# Your IBM Cloud API Key
IBM_CLOUD_API_KEY=abc123xyz...your_real_key_here

# Your watsonx.ai Project ID (a UUID like 12345678-abcd-...)
WATSONX_PROJECT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Regional endpoint — change if your project is in a different region
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com

# Flask session secret — generate a strong value with the command below
FLASK_SECRET_KEY=your_random_secret_here

# Set to "production" when deploying
FLASK_ENV=development
```

### Generating a strong Flask secret key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output and paste it as the value for `FLASK_SECRET_KEY`.

### IBM Cloud regional URLs

| Region | URL |
|---|---|
| Dallas (us-south) — **default** | `https://us-south.ml.cloud.ibm.com` |
| Frankfurt (eu-de) | `https://eu-de.ml.cloud.ibm.com` |
| London (eu-gb) | `https://eu-gb.ml.cloud.ibm.com` |
| Tokyo (jp-tok) | `https://jp-tok.ml.cloud.ibm.com` |

---

## ▶️ Running the App

### Development server (recommended for local use)

```bash
flask run
```

Or equivalently:

```bash
python app.py
```

The app starts at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

To allow access from other devices on your network:

```bash
flask run --host=0.0.0.0 --port=5000
```

### Confirm it's running

Open your browser and navigate to `http://127.0.0.1:5000`. You should see the **Chef Aanya** welcome screen with the four-tab interface.

---

## ✨ Application Features

### 1 — 💬 Conversational Chat (Chat tab)
- Multi-turn conversation with memory (last 20 exchanges retained per session)
- Typing indicator animation while the AI is generating a response
- Inline recipe preview cards appear alongside AI replies
- Quick-prompt buttons on the welcome screen for instant inspiration
- Supports `Shift+Enter` for multi-line messages

### 2 — 🛒 Ingredient Inventory Dashboard (Pantry tab)
- Type one or more comma-separated ingredients and press **Enter** or **Add**
- One-click **Quick-Add** buttons for 20 common pantry staples
- Remove individual ingredients with the × tag button
- **Clear all** button to reset the pantry instantly
- Pantry is persisted across tabs in the Flask session

### 3 — 🍽️ Recipe Recommendations Panel (Recipes tab)
- RAG-ranked cards sorted by cosine similarity to your pantry
- Each card shows: cook time, cuisine, ingredient match % bar, matched (green) vs missing (amber) ingredients, dietary tags, and a food-waste score badge (🌿 High / ♻️ Medium / ⚠️ Low)
- Click any card to jump to Chat and ask Chef Aanya to walk you through that recipe
- **Refresh** button to re-rank after updating the pantry

### 4 — ❤️ Dietary & Substitution Planner (Dietary tab)
- Toggle restriction cards: Vegetarian, Vegan, Gluten-Free, High-Protein, Quick (≤20 min)
- Allergy awareness cards: Nut Allergy, Dairy-Free, Egg-Free
- Free-text substitution input — describe what you want to replace and Chef Aanya will respond in the chat
- Active filters are shown in the sidebar and applied to all RAG retrievals

### 5 — 🌙 Dark Mode
- Toggle in the top-right header — persists for the browser session
- Full dark colour palette with zero flicker

### 6 — 📱 Mobile Responsive
- Sidebar hidden on screens below 900 px
- Fixed bottom navigation bar on mobile with tab icons
- All cards and inputs adapt to small screens via Bootstrap 5 grid

---

## 🔌 REST API Reference

All endpoints are served by Flask. The frontend communicates via `fetch()` JSON calls.

| Method | Endpoint | Request Body | Response |
|---|---|---|---|
| `GET` | `/` | — | Renders `index.html` |
| `POST` | `/api/chat` | `{ message, ingredients[], filters[] }` | `{ reply, recipes[], history[] }` |
| `GET` | `/api/ingredients` | — | `{ ingredients[] }` |
| `POST` | `/api/ingredients` | `{ ingredients[] }` | `{ ingredients[] }` (merged) |
| `DELETE` | `/api/ingredients` | `{ ingredient }` | `{ ingredients[] }` |
| `GET` | `/api/filters` | — | `{ filters[] }` |
| `POST` | `/api/filters` | `{ filters[] }` | `{ filters[] }` |
| `GET` | `/api/recipes` | — | `{ recipes[] }` (RAG-ranked) |
| `POST` | `/api/session/clear` | — | `{ status: "cleared" }` |

### Example — POST `/api/chat`

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What can I cook tonight?",
    "ingredients": ["onion", "tomato", "lentils", "garlic"],
    "filters": ["vegan"]
  }'
```

---

## 🔍 RAG System Explained

The Retrieval-Augmented Generation pipeline lives entirely in `app.py` with no external vector database required.

```
User Pantry (list of strings)
        │
        ▼
TF-IDF Query Vector  ←──  TfidfVectorizer fitted on RECIPE_CORPUS
        │
        ▼
Cosine Similarity  ──→  Ranked recipe indices
        │
        ▼
Dietary Tag Filter  ──→  Filtered candidates
        │
        ▼
Top-K Recipes  (match %, matched[], missing[])
        │
        ▼
Injected into Granite Prompt  ──→  watsonx.ai API
        │
        ▼
AI Response with substitutions, tips, waste score
```

**Key functions:**

| Function | File | Purpose |
|---|---|---|
| `_build_vector_store()` | `app.py:345` | Fits TF-IDF matrix at startup over all 15 recipes |
| `retrieve_recipes()` | `app.py:359` | Cosine similarity retrieval + tag filtering + match/missing annotation |
| `_build_system_prompt()` | `app.py:410` | Assembles the Granite system prompt from `AGENT_INSTRUCTIONS` |
| `_build_user_message()` | `app.py:422` | Injects pantry, filters, RAG context, and history into the user turn |
| `call_watsonx()` | `app.py:462` | Sends the full `<\|system\|>…<\|user\|>…<\|assistant\|>` prompt to Granite |

---

## 🎛️ Customising the Agent

Open [`app.py`](app.py) and find the `AGENT_INSTRUCTIONS` dictionary at the top of the file (line 24). Each key can be edited freely — no other code needs to change.

```python
AGENT_INSTRUCTIONS = {
    "persona":        "...",   # Name, personality, verbosity
    "cuisine_focus":  "...",   # Regional cuisines, spice systems
    "safety_rules":   "...",   # Allergen handling, cross-contamination warnings
    "waste_reduction":"...",   # Food-waste scoring philosophy
    "portions":       "...",   # Default serving size and scaling rules
    "format_rules":   "...",   # Recipe output structure and markdown style
}
```

### Example customisations

**Change the agent's name and focus to Italian cuisine:**
```python
"persona": (
    "You are Chef Marco, a passionate Italian home-cooking expert. "
    "You speak with warmth and enthusiasm about fresh, seasonal ingredients."
),
"cuisine_focus": (
    "You specialise in Northern and Southern Italian cuisine — pasta, risotto, "
    "pizza, antipasti, and traditional regional dishes."
),
```

**Strengthen nut allergy safety:**
```python
"safety_rules": (
    "CRITICAL: The user has a severe tree-nut and peanut allergy. "
    "NEVER suggest any recipe containing nuts or nut-derived oils. "
    "Always flag cross-contamination risks explicitly."
),
```

**Change default portion size:**
```python
"portions": "Default to 4-person servings (family size). Always state quantities clearly.",
```

---

## 📚 Extending the Recipe Dataset

Add new entries to the `RECIPE_CORPUS` list in [`app.py`](app.py) (starting at line 143). The TF-IDF index rebuilds automatically every time the app starts.

```python
{
    "id":          "r16",                          # unique string ID
    "name":        "Butter Chicken",
    "cuisine":     "Mughlai",
    "cook_time":   45,                             # minutes (integer)
    "tags":        ["non-vegetarian", "gluten-free"],
    "ingredients": [
        "chicken", "butter", "cream", "tomato",
        "onion", "garlic", "ginger", "garam masala",
        "kasuri methi", "oil", "salt",
    ],
    "description": "Rich tomato and cream-based chicken curry.",
    "waste_score": 6,                              # 1 (low) to 10 (high)
},
```

**`waste_score` guide:**

| Score | Meaning |
|---|---|
| 9–10 | Uses up highly perishable items (overripe fruit, wilting greens, stale bread, leftover cooked rice) |
| 6–8 | Standard pantry recipe with moderate waste impact |
| 1–5 | Requires mostly fresh-bought, non-perishable, or specialty ingredients |

---

## 🏷️ Dietary Filter Tags

These tag strings must match **exactly** (case-insensitive) between recipe `tags[]` entries and the filter IDs used in the UI.

| Tag | Description |
|---|---|
| `vegetarian` | No meat, poultry, or seafood |
| `vegan` | No animal products (including dairy, eggs, honey, ghee) |
| `gluten-free` | No wheat, barley, rye, semolina, or regular soy sauce |
| `high-protein` | Rich in plant or animal protein sources |
| `quick` | Total cook time ≤ 20 minutes |
| `non-vegetarian` | Contains meat, poultry, or seafood |
| `drink` | Beverage recipes |
| `sweet` | Desserts or sweet dishes |

To add a new filter to the UI, append an entry to the `DIETARY_OPTIONS` or `ALLERGY_OPTIONS` array in [`templates/index.html`](templates/index.html):

```javascript
const DIETARY_OPTIONS = [
  // ... existing entries ...
  {
    id:   "dairy-free",
    label:"Dairy-Free",
    icon: "🥛",
    desc: "Excludes all milk, cheese, cream, butter, and yogurt."
  },
];
```

Then add the matching tag string to any recipes in `RECIPE_CORPUS` that qualify.

---

## 🚀 Production Deployment

### Using Gunicorn (Linux / macOS)

```bash
# Set production environment
export FLASK_ENV=production

# Run with 4 worker processes
gunicorn app:app --workers 4 --bind 0.0.0.0:8080 --timeout 120
```

### Using Gunicorn on Windows (via WSL)

Gunicorn does not run natively on Windows. Use **Windows Subsystem for Linux (WSL)** or deploy to a Linux server / container.

### Environment variables for production

Ensure these are set in your server environment (or use a secrets manager — do **not** rely on `.env` in production):

```bash
IBM_CLOUD_API_KEY=...
WATSONX_PROJECT_ID=...
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=...   # long random string
FLASK_ENV=production
```

### Recommended production stack

```
Browser → nginx (reverse proxy) → Gunicorn → Flask app
```

Example minimal nginx config block:

```nginx
location / {
    proxy_pass         http://127.0.0.1:8080;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_read_timeout 120s;
}
```

---

## 🛠️ Troubleshooting

### `ModuleNotFoundError: No module named 'flask'`
The virtual environment is not active, or dependencies were not installed.
```bash
# Re-activate the venv
.venv\Scripts\Activate.ps1       # Windows
source .venv/bin/activate        # macOS / Linux

# Re-install dependencies
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'ibm_watsonx_ai'`
```bash
pip install "ibm-watsonx-ai>=1.0.0"
```

### App starts but shows "Demo mode" responses
Your `.env` file is either missing, the `IBM_CLOUD_API_KEY` is blank, or the `WATSONX_PROJECT_ID` is blank. Verify:
```bash
# Check .env exists
ls -la .env          # macOS/Linux
Get-Item .env        # Windows PowerShell
```
Ensure both values are filled in and the file is saved.

### `401 Unauthorized` from watsonx.ai
- The API key is incorrect or has been revoked.
- Regenerate it at [cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys).

### `404` from watsonx.ai
- The `WATSONX_PROJECT_ID` is wrong, or the project does not exist in the selected region.
- Double-check the Project ID in your watsonx.ai project under **Manage → General**.
- Confirm `IBM_CLOUD_URL` matches the region where the project was created.

### `Address already in use` on port 5000
Another process is using port 5000 (common on macOS where AirPlay uses 5000).
```bash
flask run --port=5001
# Then open http://127.0.0.1:5001
```

### Recipes tab shows "Add ingredients to your pantry…"
No ingredients have been added yet, or the session expired. Open the **Pantry** tab, add some ingredients, then return to **Recipes** and click **Refresh**.

### `pip` is not recognized / `pip command not found`

This is the most common Windows issue — Python was installed without adding it to PATH,
or the terminal window was not restarted after installation.

**Fix A — Restart your terminal** (most common fix):
Close PowerShell completely, reopen it, then run `pip --version` again.

**Fix B — Use the module form** (always works, no PATH needed):
```powershell
python -m pip install -r requirements.txt
```

**Fix C — Python itself isn't installed yet** (see Step 2 above for the silent-install commands):
```powershell
# Quick check — if this prints "Python was not found..." then Python is missing:
python --version
```

**Fix D — Windows Store stub is blocking the real Python:**
Go to **Settings → Apps → Advanced app settings → App execution aliases** and turn **OFF**
both `python.exe` and `python3.exe` store entries, then restart PowerShell.

---

## 🔒 Security Notes

- **Never commit `.env`** to version control. Add it to `.gitignore`:
  ```
  .env
  .venv/
  __pycache__/
  *.pyc
  ```
- The `FLASK_SECRET_KEY` signs session cookies. Use a minimum 32-character random string in production.
- The IBM Cloud API Key grants access to your IBM Cloud account. Rotate it immediately if exposed.
- Flask's built-in development server (`flask run`) is **not suitable for production** — use Gunicorn behind nginx.
- Session data (ingredients, chat history) is stored in the server-side Flask session cookie. No data is persisted to disk or a database between server restarts.

---

## 📄 Licence

This project is released for educational and demonstration purposes.
IBM watsonx.ai usage is subject to [IBM's terms of service](https://www.ibm.com/legal).

---

*Made with ❤️ using IBM watsonx.ai · Chef Aanya powered by IBM Granite 3.3 8B Instruct*
