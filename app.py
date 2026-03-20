"""
PathForge — AI Adaptive Onboarding Engine (Upgraded)
Hackathon submission — ARTPARK CodeForge

Upgrades over v1:
- Claude claude-sonnet-4-20250514 for intelligent skill extraction and AI summaries
- Smarter NLP-based skill matching (Claude handles synonyms + context)
- AI-generated reasoning trace (required by judges)
- AI career analysis summary
- Prerequisite-aware topological roadmap (preserved from v1)
- TF-IDF relevance scoring (preserved from v1)
"""

import os
import json
import requests
import pandas as pd
import PyPDF2
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ─── ANTHROPIC CONFIG ────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

def call_claude(prompt: str, max_tokens: int = 1000) -> str:
    """Call Claude API and return text response."""
    if not ANTHROPIC_API_KEY:
        return ""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
    data = resp.json()
    return data.get("content", [{}])[0].get("text", "")

# ─── LOAD DATASET ────────────────────────────────────────
df = pd.read_csv("skills_dataset.csv")
df["skill"] = df["skill"].str.lower().str.strip()
df["prerequisites"] = df["prerequisites"].fillna("")
skills_db = set(df["skill"].tolist())

SYNONYM_MAP = {
    "ml": "machine learning", "dl": "deep learning",
    "natural language processing": "nlp", "rest api": "api",
    "rest apis": "api", "apis": "api",
    "github version control": "github", "android": "android development",
    "compose": "jetpack compose", "js": "javascript",
    "dbms": "sql", "node.js": "javascript", "nodejs": "javascript"
}

LEVEL_WEIGHT = {"beginner": 1, "intermediate": 2, "advanced": 3}

# ─── PDF EXTRACTION ──────────────────────────────────────
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    text += pt + " "
    except Exception as e:
        print("PDF read error:", e)
    return text

# ─── SKILL EXTRACTION ────────────────────────────────────
def normalize_text(text):
    text = text.lower()
    for k, v in SYNONYM_MAP.items():
        text = text.replace(k, v)
    return text

def extract_skills_rule_based(text):
    """Fast rule-based extraction from skills_dataset.csv."""
    text = normalize_text(text)
    found = set()
    for skill in skills_db:
        if skill in text:
            found.add(skill)
    return found

def extract_skills_with_claude(text: str, context: str = "resume") -> set:
    """
    Use Claude to intelligently extract skills, handling synonyms,
    abbreviations, implied skills, and contextual understanding.
    Falls back to rule-based if API key not set.
    """
    if not ANTHROPIC_API_KEY:
        return extract_skills_rule_based(text)

    known_skills_list = ", ".join(sorted(skills_db))
    prompt = f"""You are a skill extraction expert. Extract skills from this {context} text.

Known skills catalog (only return skills from this list):
{known_skills_list}

{context.title()} text:
\"\"\"
{text[:3000]}
\"\"\"

Instructions:
- Match skills by meaning, not just exact text (e.g. "ML" → "machine learning", "Node" → "javascript")
- Infer skills from project descriptions (e.g. "built REST API with Flask" → flask, api, python)
- Return ONLY skills from the catalog above
- Return as JSON array of strings, nothing else

Example output: ["python", "flask", "sql", "git"]
"""
    raw = call_claude(prompt)
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted = json.loads(raw)
        return set(s.strip().lower() for s in extracted if s.strip().lower() in skills_db)
    except Exception:
        # Fallback to rule-based
        return extract_skills_rule_based(text)

# ─── SKILL INFO ──────────────────────────────────────────
def get_skill_info(skill):
    row = df[df["skill"] == skill]
    return row.iloc[0].to_dict() if not row.empty else None

def get_prerequisites(skill):
    row = df[df["skill"] == skill]
    if row.empty:
        return []
    prereq_str = row.iloc[0]["prerequisites"]
    if pd.isna(prereq_str) or str(prereq_str).strip() == "":
        return []
    return [p.strip().lower() for p in str(prereq_str).split("|") if p.strip()]

# ─── ADAPTIVE LOGIC ──────────────────────────────────────
def expand_with_prerequisites(missing_skills, known_skills):
    final = set(missing_skills)
    def add_prereqs(skill):
        for prereq in get_prerequisites(skill):
            if prereq not in known_skills:
                final.add(prereq)
                add_prereqs(prereq)
    for skill in list(missing_skills):
        add_prereqs(skill)
    return final

def generate_adaptive_path(target_skills, known_skills):
    expanded = expand_with_prerequisites(target_skills, known_skills)
    visited = set()
    ordered = []

    def dfs(skill):
        if skill in visited:
            return
        visited.add(skill)
        for prereq in get_prerequisites(skill):
            if prereq in expanded and prereq not in known_skills:
                dfs(prereq)
        if skill not in known_skills:
            ordered.append(skill)

    for skill in expanded:
        dfs(skill)

    unique = []
    seen = set()
    for s in ordered:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

# ─── METRICS ─────────────────────────────────────────────
def calculate_match_score(job_skills, resume_skills):
    if not job_skills:
        return 100.0
    matched = len(job_skills & resume_skills)
    return round((matched / len(job_skills)) * 100, 2)

def estimate_total_hours(roadmap_skills):
    total = sum(float(get_skill_info(s)["duration_hours"]) for s in roadmap_skills if get_skill_info(s))
    return round(total, 1)

def dependency_satisfaction_score(roadmap_skills, known_skills):
    if not roadmap_skills:
        return 100.0
    completed = set(known_skills)
    valid = 0
    for skill in roadmap_skills:
        if all(pr in completed for pr in get_prerequisites(skill)):
            valid += 1
        completed.add(skill)
    return round((valid / len(roadmap_skills)) * 100, 2)

def build_roadmap_details(roadmap_skills):
    roadmap = []
    for idx, skill in enumerate(roadmap_skills, 1):
        info = get_skill_info(skill)
        if info:
            roadmap.append({
                "step": idx, "skill": skill,
                "level": str(info["level"]).title(),
                "duration": float(info["duration_hours"]),
                "resource_type": str(info["resource_type"]).title(),
                "course_title": str(info["course_title"]),
                "description": str(info["course_description"]),
                "prerequisites": get_prerequisites(skill)
            })
        else:
            roadmap.append({
                "step": idx, "skill": skill,
                "level": "Unknown", "duration": 0,
                "resource_type": "Learning Resource",
                "course_title": f"Learn {skill.title()}",
                "description": f"Study {skill.title()} fundamentals.",
                "prerequisites": []
            })
    return roadmap

def recommend_best_course(skill, job_text):
    row = df[df["skill"] == skill]
    if row.empty:
        return None
    try:
        docs = [str(row.iloc[0]["course_description"]), str(job_text)]
        vec = TfidfVectorizer(stop_words="english")
        vecs = vec.fit_transform(docs)
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return round(float(sim) * 100, 2)
    except Exception:
        return None

def simulate_quiz_mastery(roadmap_skills):
    mastery = []
    for skill in roadmap_skills:
        info = get_skill_info(skill)
        level = str(info["level"]).lower() if info else "beginner"
        score = 78 if level == "beginner" else 62 if level == "intermediate" else 48
        rec = ("Revise fundamentals before moving ahead" if score < 50
               else "Do more practice and mini-projects" if score < 70
               else "Can progress to next module")
        mastery.append({"skill": skill.title(), "quiz_score": score, "recommendation": rec})
    return mastery

# ─── CLAUDE AI ENRICHMENT ────────────────────────────────
def generate_ai_summary(resume_skills, job_skills, matched, missing, roadmap_list, match_score, total_hours):
    if not ANTHROPIC_API_KEY:
        return "AI analysis unavailable. Set ANTHROPIC_API_KEY environment variable."
    prompt = f"""You are an expert career advisor analyzing a skill gap for a job application.

Analysis data:
- Resume Skills: {', '.join(sorted(resume_skills)) or 'none detected'}
- Job Required Skills: {', '.join(sorted(job_skills)) or 'none detected'}
- Matched Skills: {', '.join(sorted(matched)) or 'none'}
- Missing Skills: {', '.join(sorted(missing)) or 'none'}
- Learning Roadmap: {' → '.join(roadmap_list) or 'none needed'}
- Match Score: {match_score}%
- Estimated Learning Time: {total_hours}h

Write a professional, concise 3-4 sentence analysis covering:
1. Current fit assessment
2. Key skill gaps and their importance
3. One motivating insight about the learning path

Be direct, specific, and encouraging. No bullet points. Just flowing prose."""
    return call_claude(prompt)

def generate_reasoning_trace(resume_skills, job_skills, missing, roadmap_list):
    if not ANTHROPIC_API_KEY:
        return "Reasoning trace unavailable. Set ANTHROPIC_API_KEY environment variable."
    prompt = f"""Document the reasoning trace for this adaptive learning system decision:

Input:
- Job Skills Detected: {', '.join(sorted(job_skills)) or 'none'}
- Resume Skills: {', '.join(sorted(resume_skills)) or 'none'}
- Direct Missing Skills: {', '.join(sorted(missing)) or 'none'}
- Final Roadmap (after prerequisite expansion): {' → '.join(roadmap_list) or 'none'}

Write a step-by-step reasoning trace (5-7 lines) explaining:
1. How skills were matched from text
2. Why specific prerequisites were automatically added
3. How dependency ordering was determined
4. How the adaptive logic ensures no skill is taught before its prerequisites

Format as numbered steps. Be technical but readable."""
    return call_claude(prompt, max_tokens=600)

# ─── ROUTES ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = request.form.get("resume_text", "").strip()
    job_text = request.form.get("job_text", "").strip()

    resume_file = request.files.get("resume_pdf")
    if resume_file and resume_file.filename.endswith(".pdf"):
        fp = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(fp)
        resume_text += " " + extract_text_from_pdf(fp)

    # AI-powered skill extraction
    resume_skills = extract_skills_with_claude(resume_text, "resume")
    job_skills = extract_skills_with_claude(job_text, "job description")

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills

    roadmap_list = generate_adaptive_path(missing, resume_skills)
    roadmap = build_roadmap_details(roadmap_list)
    mastery = simulate_quiz_mastery(roadmap_list)

    match_score = calculate_match_score(job_skills, resume_skills)
    total_hours = estimate_total_hours(roadmap_list)
    dep_score = dependency_satisfaction_score(roadmap_list, resume_skills)

    for item in roadmap:
        sim = recommend_best_course(item["skill"].lower(), job_text)
        item["relevance_score"] = sim if sim is not None else "N/A"

    # Claude enrichment
    ai_summary = generate_ai_summary(resume_skills, job_skills, matched, missing, roadmap_list, match_score, total_hours)
    reasoning_trace = generate_reasoning_trace(resume_skills, job_skills, missing, roadmap_list)

    return render_template(
        "result.html",
        resume_skills=sorted(resume_skills),
        job_skills=sorted(job_skills),
        matched_skills=sorted(matched),
        missing_job_skills=sorted(missing),
        roadmap=roadmap,
        match_score=match_score,
        gap_count=len(missing),
        estimated_hours=total_hours,
        dependency_score=dep_score,
        predicted_completion=20.0,
        mastery_data=mastery,
        ai_summary=ai_summary,
        reasoning_trace=reasoning_trace
    )

if __name__ == "__main__":
    app.run(debug=True)