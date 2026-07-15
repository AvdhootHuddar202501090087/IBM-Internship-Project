"""
Recipe Preparation Agent — IBM watsonx.ai + Flask
===================================================
Entry point:  flask run   (or  gunicorn app:app  for production)
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# AGENT INSTRUCTIONS  ← customise freely without touching any other code
# ─────────────────────────────────────────────────────────────────────────────
AGENT_INSTRUCTIONS = {
    # --- Personality & Tone ------------------------------------------------
    "persona": (
        "You are Chef Aanya, a warm, encouraging AI culinary assistant "
        "specialising in home cooking, Indian cuisine, and zero-waste cooking. "
        "You speak in a friendly, conversational tone with occasional culinary "
        "enthusiasm (e.g. 'That's a lovely combination!'). "
        "Keep responses concise—no more than 4-5 short paragraphs unless a "
        "detailed step-by-step recipe is explicitly requested."
    ),

    # --- Culinary Specialisation -------------------------------------------
    "cuisine_focus": (
        "You have deep knowledge of Indian regional cuisine "
        "(North Indian, South Indian, Bengali, Gujarati, Punjabi, Kerala, "
        "Rajasthani, and street food). You naturally suggest Indian spice blends "
        "(garam masala, chaat masala, sambar powder, etc.) as enhancements. "
        "You are equally comfortable with global cuisines such as Italian, "
        "Mediterranean, East Asian, and Mexican."
    ),

    # --- Dietary & Allergen Safety Rules -----------------------------------
    "safety_rules": (
        "CRITICAL: Always acknowledge user-reported allergies (nuts, dairy, "
        "gluten, shellfish, soy, eggs) before suggesting any recipe. "
        "Never suggest an allergen without a clearly labelled substitution. "
        "When a user selects Vegan, omit ALL animal products including ghee, "
        "honey, and paneer. For Gluten-Free, avoid atta, maida, semolina (sooji), "
        "and regular soy sauce. Always mention cross-contamination risks for "
        "nut or gluten allergies."
    ),

    # --- Food-Waste & Substitution Philosophy ------------------------------
    "waste_reduction": (
        "Your primary goal is to help users cook with what they already have "
        "and reduce household food waste. When ingredients are missing, first "
        "propose a reasonable pantry substitute before suggesting a shopping trip. "
        "Highlight 'use-it-up' opportunities for wilting vegetables, overripe "
        "fruit, or stale bread. Always award a food-waste impact score (1-10) "
        "where 10 = uses up the most perishables."
    ),

    # --- Portion & Scaling -------------------------------------------------
    "portions": (
        "Default to 2-person servings unless the user specifies otherwise. "
        "When asked to scale, adjust every ingredient quantity proportionally "
        "and note any technique changes needed (e.g., pan size, cooking time)."
    ),

    # --- Response Formatting -----------------------------------------------
    "format_rules": (
        "For recipes, always use this structure: "
        "1) Recipe name & emoji  "
        "2) Prep / Cook / Total time  "
        "3) Ingredients (with quantities)  "
        "4) Step-by-step Method  "
        "5) Chef's Tip (a practical, non-obvious tip)  "
        "6) Substitutions for missing items  "
        "7) Food-waste score with a brief justification. "
        "Use markdown bold (**) for section headers and bullet points for lists."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Environment & App bootstrap
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)

IBM_CLOUD_API_KEY   = os.getenv("IBM_CLOUD_API_KEY", "")
WATSONX_PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID", "")
IBM_CLOUD_URL       = os.getenv("IBM_CLOUD_URL", "https://us-south.ml.cloud.ibm.com")

# ─────────────────────────────────────────────────────────────────────────────
# watsonx.ai client (lazy-initialised once, reused across requests)
# ─────────────────────────────────────────────────────────────────────────────
_wx_client: Any = None

def _get_wx_client():
    """Return a cached watsonx.ai ModelInference client."""
    global _wx_client
    if _wx_client is not None:
        return _wx_client

    if not IBM_CLOUD_API_KEY or not WATSONX_PROJECT_ID:
        return None                             # run in demo / mock mode

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

        creds  = Credentials(url=IBM_CLOUD_URL, api_key=IBM_CLOUD_API_KEY)
        client = APIClient(creds)

        _wx_client = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            api_client=client,
            project_id=WATSONX_PROJECT_ID,
            params={
                Params.MAX_NEW_TOKENS: 1200,
                Params.TEMPERATURE:    0.7,
                Params.TOP_P:          0.95,
                Params.REPETITION_PENALTY: 1.1,
            },
        )
        return _wx_client

    except Exception as exc:
        print(f"[watsonx] init error — running in mock mode: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Starter Recipe Dataset  (mock vector store / RAG corpus)
# ─────────────────────────────────────────────────────────────────────────────
RECIPE_CORPUS: list[dict] = [
    {
        "id": "r01",
        "name": "Dal Tadka",
        "cuisine": "North Indian",
        "cook_time": 30,
        "tags": ["vegetarian", "vegan", "gluten-free", "high-protein"],
        "ingredients": [
            "red lentils", "onion", "tomato", "garlic", "ginger",
            "cumin seeds", "mustard seeds", "turmeric", "red chili", "oil", "salt",
        ],
        "description": "Comforting split red lentils tempered with cumin and mustard seeds.",
        "waste_score": 8,
    },
    {
        "id": "r02",
        "name": "Aloo Gobi",
        "cuisine": "Punjabi",
        "cook_time": 25,
        "tags": ["vegetarian", "vegan", "gluten-free"],
        "ingredients": [
            "potato", "cauliflower", "onion", "tomato", "garlic", "ginger",
            "cumin seeds", "turmeric", "coriander powder", "garam masala", "oil", "salt",
        ],
        "description": "Dry-spiced potato and cauliflower stir-fry — a classic Punjabi favourite.",
        "waste_score": 9,
    },
    {
        "id": "r03",
        "name": "Palak Paneer",
        "cuisine": "North Indian",
        "cook_time": 35,
        "tags": ["vegetarian", "gluten-free"],
        "ingredients": [
            "spinach", "paneer", "onion", "tomato", "garlic", "ginger",
            "cream", "garam masala", "cumin seeds", "turmeric", "oil", "salt",
        ],
        "description": "Creamy spinach gravy with soft cubes of fresh cottage cheese.",
        "waste_score": 7,
    },
    {
        "id": "r04",
        "name": "Chicken Biryani",
        "cuisine": "Mughlai",
        "cook_time": 60,
        "tags": ["non-vegetarian", "gluten-free"],
        "ingredients": [
            "basmati rice", "chicken", "onion", "yogurt", "garlic", "ginger",
            "biryani masala", "saffron", "mint", "ghee", "oil", "salt",
        ],
        "description": "Aromatic layered rice and chicken slow-cooked with saffron and mint.",
        "waste_score": 6,
    },
    {
        "id": "r05",
        "name": "Masala Omelette",
        "cuisine": "Indian Street Food",
        "cook_time": 10,
        "tags": ["vegetarian", "gluten-free", "quick"],
        "ingredients": [
            "eggs", "onion", "tomato", "green chili", "coriander leaves",
            "turmeric", "oil", "salt", "pepper",
        ],
        "description": "Spicy Indian-style omelette bursting with fresh aromatics.",
        "waste_score": 9,
    },
    {
        "id": "r06",
        "name": "Vegetable Upma",
        "cuisine": "South Indian",
        "cook_time": 20,
        "tags": ["vegetarian", "vegan"],
        "ingredients": [
            "semolina", "onion", "tomato", "carrot", "peas", "mustard seeds",
            "curry leaves", "green chili", "ginger", "oil", "salt",
        ],
        "description": "Light semolina porridge with sautéed vegetables — perfect breakfast.",
        "waste_score": 8,
    },
    {
        "id": "r07",
        "name": "Chana Masala",
        "cuisine": "Punjabi",
        "cook_time": 40,
        "tags": ["vegetarian", "vegan", "gluten-free", "high-protein"],
        "ingredients": [
            "chickpeas", "onion", "tomato", "garlic", "ginger",
            "chana masala spice mix", "cumin seeds", "turmeric", "oil", "salt",
        ],
        "description": "Hearty spiced chickpea curry loaded with plant protein.",
        "waste_score": 9,
    },
    {
        "id": "r08",
        "name": "Egg Fried Rice",
        "cuisine": "Indo-Chinese",
        "cook_time": 15,
        "tags": ["vegetarian", "quick"],
        "ingredients": [
            "cooked rice", "eggs", "carrot", "spring onion", "garlic",
            "soy sauce", "black pepper", "oil", "salt",
        ],
        "description": "Quick wok-tossed fried rice — great for leftover rice.",
        "waste_score": 10,
    },
    {
        "id": "r09",
        "name": "Tomato Soup",
        "cuisine": "Continental",
        "cook_time": 25,
        "tags": ["vegetarian", "vegan", "gluten-free"],
        "ingredients": [
            "tomato", "onion", "garlic", "vegetable stock", "basil",
            "black pepper", "oil", "salt",
        ],
        "description": "Smooth roasted tomato soup with a hint of basil.",
        "waste_score": 9,
    },
    {
        "id": "r10",
        "name": "Rajma Chawal",
        "cuisine": "North Indian",
        "cook_time": 50,
        "tags": ["vegetarian", "vegan", "gluten-free", "high-protein"],
        "ingredients": [
            "kidney beans", "basmati rice", "onion", "tomato", "garlic", "ginger",
            "cumin seeds", "garam masala", "turmeric", "oil", "salt",
        ],
        "description": "Slow-simmered red kidney bean curry served over fluffy basmati rice.",
        "waste_score": 8,
    },
    {
        "id": "r11",
        "name": "Banana Pancakes",
        "cuisine": "Global",
        "cook_time": 15,
        "tags": ["vegetarian", "quick", "sweet"],
        "ingredients": [
            "overripe banana", "eggs", "flour", "milk", "baking powder",
            "cinnamon", "honey", "butter", "salt",
        ],
        "description": "Fluffy pancakes — the best use for overripe bananas.",
        "waste_score": 10,
    },
    {
        "id": "r12",
        "name": "Stale Bread Bruschetta",
        "cuisine": "Italian",
        "cook_time": 10,
        "tags": ["vegetarian", "vegan", "quick"],
        "ingredients": [
            "stale bread", "tomato", "garlic", "basil", "olive oil", "salt", "pepper",
        ],
        "description": "Transform day-old bread into crunchy bruschetta topped with fresh tomato.",
        "waste_score": 10,
    },
    {
        "id": "r13",
        "name": "Saag Aloo",
        "cuisine": "North Indian",
        "cook_time": 30,
        "tags": ["vegetarian", "vegan", "gluten-free"],
        "ingredients": [
            "spinach", "potato", "garlic", "ginger", "onion", "cumin seeds",
            "mustard seeds", "turmeric", "oil", "salt",
        ],
        "description": "Wilted spinach with spiced golden potatoes — uses up leafy greens.",
        "waste_score": 9,
    },
    {
        "id": "r14",
        "name": "Mango Lassi",
        "cuisine": "Indian",
        "cook_time": 5,
        "tags": ["vegetarian", "gluten-free", "drink"],
        "ingredients": [
            "mango", "yogurt", "milk", "sugar", "cardamom",
        ],
        "description": "Refreshing blended mango and yogurt drink spiced with cardamom.",
        "waste_score": 8,
    },
    {
        "id": "r15",
        "name": "Lemon Rice",
        "cuisine": "South Indian",
        "cook_time": 15,
        "tags": ["vegetarian", "vegan", "gluten-free", "quick"],
        "ingredients": [
            "cooked rice", "lemon", "mustard seeds", "curry leaves",
            "peanuts", "turmeric", "green chili", "oil", "salt",
        ],
        "description": "Tangy tempered rice with peanuts — ideal for leftover cooked rice.",
        "waste_score": 10,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Vector Store  (lightweight RAG)
# ─────────────────────────────────────────────────────────────────────────────
_vectorizer: TfidfVectorizer | None = None
_recipe_matrix = None

def _build_vector_store() -> None:
    """Build a TF-IDF matrix over recipe ingredient lists."""
    global _vectorizer, _recipe_matrix
    corpus_docs = [
        " ".join(r["ingredients"]) + " " + r["name"] + " " + r["description"]
        for r in RECIPE_CORPUS
    ]
    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    _recipe_matrix = _vectorizer.fit_transform(corpus_docs)


_build_vector_store()


def retrieve_recipes(
    user_ingredients: list[str],
    dietary_filters: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    RAG retrieval: rank recipes by cosine similarity to user ingredient list,
    then compute match/missing sets and annotate each candidate.
    """
    if not user_ingredients:
        return []

    query = " ".join(user_ingredients)
    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _recipe_matrix).flatten()

    # Apply dietary tag filters
    filtered_indices = []
    for idx, recipe in enumerate(RECIPE_CORPUS):
        if dietary_filters:
            tags_lower = [t.lower() for t in recipe["tags"]]
            if not all(f.lower() in tags_lower for f in dietary_filters):
                continue
        filtered_indices.append(idx)

    # Rank filtered recipes by similarity
    scored = sorted(filtered_indices, key=lambda i: sims[i], reverse=True)[:top_k]

    results = []
    user_set = {ing.lower().strip() for ing in user_ingredients}
    for idx in scored:
        recipe = RECIPE_CORPUS[idx]
        rec_set = {ing.lower().strip() for ing in recipe["ingredients"]}
        matched  = sorted(user_set & rec_set)
        missing  = sorted(rec_set - user_set)
        match_pct = math.floor(len(matched) / max(len(rec_set), 1) * 100)

        results.append({
            **recipe,
            "similarity":  round(float(sims[idx]), 3),
            "match_pct":   match_pct,
            "matched":     matched,
            "missing":     missing,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_system_prompt() -> str:
    ai = AGENT_INSTRUCTIONS
    return (
        f"{ai['persona']}\n\n"
        f"Cuisine expertise: {ai['cuisine_focus']}\n\n"
        f"Safety & allergens: {ai['safety_rules']}\n\n"
        f"Food-waste philosophy: {ai['waste_reduction']}\n\n"
        f"Portions: {ai['portions']}\n\n"
        f"Response format: {ai['format_rules']}"
    )


def _build_user_message(
    user_message: str,
    ingredients: list[str],
    dietary_filters: list[str],
    retrieved_recipes: list[dict],
    history: list[dict],
) -> str:
    parts: list[str] = []

    if ingredients:
        parts.append(f"**My available ingredients:** {', '.join(ingredients)}")

    if dietary_filters:
        parts.append(f"**Active dietary filters:** {', '.join(dietary_filters)}")

    if retrieved_recipes:
        recipe_ctx = []
        for r in retrieved_recipes[:3]:
            recipe_ctx.append(
                f"- {r['name']} ({r['cuisine']}, {r['cook_time']} min | "
                f"match {r['match_pct']}% | waste score {r['waste_score']}/10) | "
                f"Matched: {', '.join(r['matched']) or 'none'} | "
                f"Missing: {', '.join(r['missing']) or 'none'}"
            )
        parts.append("**RAG-retrieved candidate recipes:**\n" + "\n".join(recipe_ctx))

    if history:
        hist_lines = []
        for turn in history[-6:]:          # last 3 exchanges for context window
            role = "User" if turn["role"] == "user" else "Chef Aanya"
            hist_lines.append(f"{role}: {turn['content']}")
        parts.append("**Recent conversation:**\n" + "\n".join(hist_lines))

    parts.append(f"**User's current message:** {user_message}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# watsonx.ai inference (with mock fallback)
# ─────────────────────────────────────────────────────────────────────────────
def call_watsonx(
    user_message: str,
    ingredients: list[str],
    dietary_filters: list[str],
    retrieved_recipes: list[dict],
    history: list[dict],
) -> str:
    system_prompt  = _build_system_prompt()
    user_turn      = _build_user_message(
        user_message, ingredients, dietary_filters, retrieved_recipes, history
    )

    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_turn}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    client = _get_wx_client()

    if client is None:
        # ── Mock / Demo response ──────────────────────────────────────────────
        if retrieved_recipes:
            top = retrieved_recipes[0]
            missing_note = (
                f" I noticed you're missing **{', '.join(top['missing'][:3])}**"
                "—I'll suggest substitutions below." if top["missing"] else ""
            )
            return (
                f"**{top['name']} 🍽️**\n\n"
                f"*{top['cuisine']} · {top['cook_time']} min · "
                f"Waste score {top['waste_score']}/10*{missing_note}\n\n"
                "*(Demo mode — connect IBM watsonx credentials in `.env` for "
                "full AI-generated recipes and substitution advice.)*"
            )
        return (
            "Hi! I'm **Chef Aanya** 👩‍🍳  Add some ingredients to your pantry "
            "and I'll suggest the best recipes to cook with what you have!\n\n"
            "*(Demo mode — add your IBM watsonx credentials to `.env` to unlock "
            "full AI responses.)*"
        )

    try:
        result = client.generate_text(prompt=full_prompt)
        return result.strip() if isinstance(result, str) else str(result).strip()
    except Exception as exc:
        return f"⚠️ watsonx.ai error: {exc}. Please check your credentials and try again."


# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
        session["history"]    = []
        session["ingredients"]= []
        session["filters"]    = []
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint — accepts a user message, runs RAG, calls watsonx."""
    data    = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # Merge any ingredient / filter updates sent with this turn
    if "ingredients" in data:
        session["ingredients"] = data["ingredients"]
    if "filters" in data:
        session["filters"] = data["filters"]

    session.modified = True

    ingredients = session.get("ingredients", [])
    filters     = session.get("filters", [])
    history     = session.get("history", [])

    # RAG retrieval
    retrieved = retrieve_recipes(ingredients, filters, top_k=5)

    # Watsonx call
    ai_response = call_watsonx(message, ingredients, filters, retrieved, history)

    # Persist conversation turn
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": ai_response})
    session["history"] = history[-40:]  # keep last 20 exchanges
    session.modified   = True

    return jsonify({
        "reply":     ai_response,
        "recipes":   retrieved,
        "history":   session["history"],
    })


@app.route("/api/ingredients", methods=["GET", "POST", "DELETE"])
def ingredients():
    """Manage the session ingredient list."""
    if request.method == "GET":
        return jsonify({"ingredients": session.get("ingredients", [])})

    if request.method == "POST":
        data  = request.get_json(force=True)
        items = data.get("ingredients", [])
        session["ingredients"] = list({
            i.lower().strip()
            for i in (session.get("ingredients", []) + items)
            if i.strip()
        })
        session.modified = True
        return jsonify({"ingredients": session["ingredients"]})

    # DELETE
    data = request.get_json(force=True)
    item = (data.get("ingredient") or "").lower().strip()
    session["ingredients"] = [
        i for i in session.get("ingredients", []) if i != item
    ]
    session.modified = True
    return jsonify({"ingredients": session["ingredients"]})


@app.route("/api/filters", methods=["GET", "POST"])
def filters():
    """Get or set dietary filters."""
    if request.method == "GET":
        return jsonify({"filters": session.get("filters", [])})

    data = request.get_json(force=True)
    session["filters"] = data.get("filters", [])
    session.modified = True
    return jsonify({"filters": session["filters"]})


@app.route("/api/recipes", methods=["GET"])
def recipes():
    """Return RAG-ranked recipes for the current ingredient list."""
    ingredients_list = session.get("ingredients", [])
    filters_list     = session.get("filters", [])
    results = retrieve_recipes(ingredients_list, filters_list, top_k=6)
    return jsonify({"recipes": results})


@app.route("/api/session/clear", methods=["POST"])
def clear_session():
    """Reset conversation history (keep ingredients & filters)."""
    session["history"] = []
    session.modified   = True
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV", "development") == "development", port=5000)
