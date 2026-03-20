from flask import Flask, render_template, request
import os
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------
# CONFIG
# ---------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_csv("skills_dataset.csv")
df["skill"] = df["skill"].str.lower().str.strip()
df["prerequisites"] = df["prerequisites"].fillna("")

# ---------------------------
# GLOBAL SKILLS DB
# ---------------------------
skills_db = set(df["skill"].tolist())

# ---------------------------
# SYNONYMS / NORMALIZATION MAP
# Helps skill extraction
# ---------------------------
synonym_map = {
    "ml": "machine learning",
    "dl": "deep learning",
    "natural language processing": "nlp",
    "rest api": "api",
    "rest apis": "api",
    "apis": "api",
    "github version control": "github",
    "android": "android development",
    "compose": "jetpack compose",
    "js": "javascript",
    "dbms": "sql"
}

# ---------------------------
# LEVEL WEIGHTS
# Used for sorting
# ---------------------------
level_weight = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3
}

# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print("PDF read error:", e)
    return text

# ---------------------------
# NORMALIZE TEXT
# ---------------------------
def normalize_text(text):
    text = text.lower()
    for k, v in synonym_map.items():
        text = text.replace(k, v)
    return text

# ---------------------------
# SKILL EXTRACTION
# Rule-based skill extraction
# ---------------------------
def extract_skills(text):
    text = normalize_text(text)
    found_skills = set()

    for skill in skills_db:
        if skill in text:
            found_skills.add(skill)

    return found_skills

# ---------------------------
# GET SKILL INFO FROM DATASET
# ---------------------------
def get_skill_info(skill):
    row = df[df["skill"] == skill]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

# ---------------------------
# GET PREREQUISITES
# ---------------------------
def get_prerequisites(skill):
    row = df[df["skill"] == skill]
    if row.empty:
        return []

    prereq_str = row.iloc[0]["prerequisites"]
    if pd.isna(prereq_str) or str(prereq_str).strip() == "":
        return []

    return [p.strip().lower() for p in str(prereq_str).split("|") if p.strip()]

# ---------------------------
# EXPAND MISSING SKILLS WITH PREREQUISITES
# This is the true adaptive logic
# ---------------------------
def expand_with_prerequisites(missing_skills, known_skills):
    final_skills = set(missing_skills)

    def add_prereqs(skill):
        prereqs = get_prerequisites(skill)
        for prereq in prereqs:
            if prereq not in known_skills:
                final_skills.add(prereq)
                add_prereqs(prereq)

    for skill in list(missing_skills):
        add_prereqs(skill)

    return final_skills

# ---------------------------
# TOPOLOGICAL / ADAPTIVE SORT
# Ensures prerequisites come before advanced skills
# ---------------------------
def generate_adaptive_path(target_skills, known_skills):
    expanded_skills = expand_with_prerequisites(target_skills, known_skills)

    visited = set()
    ordered = []

    def dfs(skill):
        if skill in visited:
            return
        visited.add(skill)

        prereqs = get_prerequisites(skill)
        for prereq in prereqs:
            if prereq in expanded_skills and prereq not in known_skills:
                dfs(prereq)

        if skill not in known_skills:
            ordered.append(skill)

    for skill in expanded_skills:
        dfs(skill)

    # Remove duplicates while preserving order
    unique_ordered = []
    seen = set()
    for skill in ordered:
        if skill not in seen:
            seen.add(skill)
            unique_ordered.append(skill)

    # Final refinement using level + duration
    def sort_key(skill):
        info = get_skill_info(skill)
        if not info:
            return (99, 999)
        return (
            level_weight.get(str(info["level"]).lower(), 99),
            float(info["duration_hours"])
        )

    # Keep dependency order, but group loosely by difficulty
    # We won't fully reorder because it may break dependency order
    return unique_ordered

# ---------------------------
# MATCH SCORE
# ---------------------------
def calculate_match_score(job_skills, resume_skills):
    if len(job_skills) == 0:
        return 100.0
    matched = len(job_skills.intersection(resume_skills))
    return round((matched / len(job_skills)) * 100, 2)

# ---------------------------
# ESTIMATED LEARNING HOURS
# ---------------------------
def estimate_total_hours(roadmap_skills):
    total = 0
    for skill in roadmap_skills:
        info = get_skill_info(skill)
        if info:
            total += float(info["duration_hours"])
    return round(total, 1)

# ---------------------------
# DEPENDENCY SATISFACTION SCORE
# Checks if roadmap order respects prerequisites
# ---------------------------
def dependency_satisfaction_score(roadmap_skills, known_skills):
    completed = set(known_skills)
    valid = 0

    for skill in roadmap_skills:
        prereqs = get_prerequisites(skill)
        if all(pr in completed for pr in prereqs):
            valid += 1
        completed.add(skill)

    if len(roadmap_skills) == 0:
        return 100.0

    return round((valid / len(roadmap_skills)) * 100, 2)

# ---------------------------
# BUILD ROADMAP DETAILS
# ---------------------------
def build_roadmap_details(roadmap_skills):
    roadmap = []

    for idx, skill in enumerate(roadmap_skills, start=1):
        info = get_skill_info(skill)

        if info:
            roadmap.append({
                "step": idx,
                "skill": skill.title(),
                "level": str(info["level"]).title(),
                "duration": float(info["duration_hours"]),
                "resource_type": str(info["resource_type"]).title(),
                "course_title": str(info["course_title"]),
                "description": str(info["course_description"]),
                "prerequisites": get_prerequisites(skill)
            })
        else:
            roadmap.append({
                "step": idx,
                "skill": skill.title(),
                "level": "Unknown",
                "duration": 0,
                "resource_type": "Learning Resource",
                "course_title": f"Learn {skill.title()}",
                "description": f"Learn the fundamentals of {skill.title()}",
                "prerequisites": []
            })

    return roadmap

# ---------------------------
# OPTIONAL TF-IDF COURSE RECOMMENDATION
# Finds the most relevant course description
# ---------------------------
def recommend_best_course_for_skill(skill, job_text):
    row = df[df["skill"] == skill]
    if row.empty:
        return None

    course_desc = row.iloc[0]["course_description"]
    docs = [str(course_desc), str(job_text)]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(docs)
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(float(sim) * 100, 2)
    except:
        return None

# ---------------------------
# QUIZ SIMULATION (Demo-friendly)
# In real version, this would come from user quizzes
# ---------------------------
def simulate_quiz_mastery(roadmap_skills):
    mastery_data = []

    for idx, skill in enumerate(roadmap_skills):
        # simple deterministic demo logic (no randomness)
        # easier skills get better score, advanced may get lower
        info = get_skill_info(skill)
        level = str(info["level"]).lower() if info else "beginner"

        if level == "beginner":
            quiz_score = 78
        elif level == "intermediate":
            quiz_score = 62
        else:
            quiz_score = 48

        if quiz_score < 50:
            recommendation = "Revise fundamentals before moving ahead"
        elif quiz_score < 70:
            recommendation = "Do more practice and mini-projects"
        else:
            recommendation = "Can progress to next module"

        mastery_data.append({
            "skill": skill.title(),
            "quiz_score": quiz_score,
            "recommendation": recommendation
        })

    return mastery_data

# ---------------------------
# OVERALL COMPLETION PREDICTION
# (Demo placeholder)
# ---------------------------
def predicted_completion_percent(roadmap_skills):
    if len(roadmap_skills) == 0:
        return 100.0
    # for demo, assume user can immediately complete first 20%
    return 20.0

# ---------------------------
# HOME PAGE
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------
# ANALYZE ROUTE
# ---------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = request.form.get("resume_text", "").strip()
    job_text = request.form.get("job_text", "").strip()

    # Handle PDF upload
    resume_file = request.files.get("resume_pdf")
    if resume_file and resume_file.filename.endswith(".pdf"):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(file_path)
        pdf_text = extract_text_from_pdf(file_path)
        resume_text = resume_text + " " + pdf_text

    # Extract skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched_skills = resume_skills.intersection(job_skills)
    missing_job_skills = job_skills - resume_skills

    # Adaptive roadmap
    roadmap_skills = generate_adaptive_path(missing_job_skills, resume_skills)

    # Metrics
    match_score = calculate_match_score(job_skills, resume_skills)
    gap_count = len(missing_job_skills)
    estimated_hours = estimate_total_hours(roadmap_skills)
    dependency_score = dependency_satisfaction_score(roadmap_skills, resume_skills)
    predicted_completion = predicted_completion_percent(roadmap_skills)

    # Detailed roadmap
    roadmap = build_roadmap_details(roadmap_skills)

    # Quiz simulation
    mastery_data = simulate_quiz_mastery(roadmap_skills)

    # TF-IDF relevance per roadmap step
    for item in roadmap:
        sim_score = recommend_best_course_for_skill(item["skill"].lower(), job_text)
        item["relevance_score"] = sim_score if sim_score is not None else "N/A"

    return render_template(
        "result.html",
        resume_skills=sorted(list(resume_skills)),
        job_skills=sorted(list(job_skills)),
        matched_skills=sorted(list(matched_skills)),
        missing_job_skills=sorted(list(missing_job_skills)),
        roadmap=roadmap,
        match_score=match_score,
        gap_count=gap_count,
        estimated_hours=estimated_hours,
        dependency_score=dependency_score,
        predicted_completion=predicted_completion,
        mastery_data=mastery_data
    )

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)