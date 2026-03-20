# ai-adaptive-onboarding-engine
# 🛤️ PathForge — AI Adaptive Onboarding Engine

> **Hackathon: ARTPARK CodeForge**  
> Upload your resume. Paste a job description. Get a personalized, prerequisite-aware learning roadmap — powered by Claude AI + TF-IDF skill-gap analysis.

---

## 🚀 What It Does

PathForge analyzes the gap between a candidate's existing skills and the requirements of a target job, then generates a structured, step-by-step learning roadmap to bridge that gap.

- **Extracts skills** from a resume (PDF or text) and a job description using Claude AI or a rule-based fallback
- **Detects the skill gap** via set difference: `job_skills − resume_skills = missing_skills`
- **Recursively expands prerequisites** — if you're missing React, it checks whether you also need JavaScript first
- **Topologically sorts** the roadmap using DFS so every prerequisite always appears before the skill that requires it
- **Scores course relevance** to the specific job description using TF-IDF cosine similarity
- **Simulates quiz-based mastery** feedback for each roadmap step
- **Surfaces out-of-catalog skills** detected by Claude that aren't yet in the course database — nothing is silently dropped

---

## 🧠 How the Adaptive Logic Works

```
Resume / Job Text
       │
       ▼
 Claude AI (or rule-based fallback)
       │  — resolves abbreviations (DSA → data structures, ML → machine learning)
       │  — infers skills from project descriptions
       │  — semantic understanding beyond keyword matching
       ▼
  Skill Gap Detection
  (job_skills − resume_skills = missing_skills)
       │
       ▼
  Prerequisite Expansion (recursive)
  + Topological DFS Sort
       │
       ▼
  TF-IDF Relevance Scoring per course
       │
       ▼
  Adaptive Mastery Simulation
       │
       ▼
  Personalized Roadmap Report
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI / Extraction | Claude API (`claude-sonnet-4-20250514`) |
| ML / Scoring | scikit-learn (TF-IDF + cosine similarity) |
| PDF Parsing | PyPDF2 |
| Data | pandas, `skills_dataset.csv` |
| Frontend | HTML, CSS (Jinja2 templates) |

---

## 📂 Project Structure

```
pathforge/
├── app.py                  # Core Flask app — all logic lives here
├── skills_dataset.csv      # Skill catalog with prerequisites, courses, durations
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # Upload form
│   └── result.html         # Adaptive roadmap report
└── uploads/                # Temporary PDF storage (auto-created)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/AgrimGangwar25/pathforge.git
cd pathforge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```




### 3. Run the app

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 🖥️ Usage

1. Open the app at `http://localhost:5000`
2. Upload your resume as a PDF **or** paste resume text
3. Paste the job description you're targeting
4. Click **Analyze** to generate your personalized roadmap

The results page shows:

- ✅ Skills you already have
- 🔴 Skills you're missing
- 🛣️ Step-by-step adaptive learning roadmap (prerequisite-ordered)
- 📊 Match score, estimated hours, dependency score
- 🧠 Quiz-based mastery simulation per skill
- 🔎 Out-of-catalog skills detected by Claude

---

## 📊 Datasets Used

The skill catalog and synonym map were designed and validated using three real-world datasets:

- **O\*NET Technology Skills DB** (`onetcenter.org`) — occupation-to-skill mapping across 923 occupations
- **Kaggle Resume Dataset** by `snehaanbhawal` — 2,484 resumes across 24 job categories
- **Kaggle Jobs & Job Description Dataset** by `kshitizregmi` — real-world JD skill validation

---

## ✨ Key Features at a Glance

- **Dual extraction mode** — Claude AI (semantic) or rule-based fallback (no API key needed)
- **60+ abbreviation synonyms** — DSA, ML, DL, NLP, k8s, CI/CD, and more
- **Recursive prerequisite graph** — learns what you need to learn first
- **TF-IDF relevance scoring** — courses ranked against your specific job description
- **Transparent reporting** — out-of-catalog skills are surfaced, not silently dropped

---


## 👤 Authors
 
**Agrim Gangwar**  
B.E. Computer Science & Business Systems — Thapar Institute of Engineering & Technology  
 [GitHub](https://github.com/AgrimGangwar25)

 **Manya Sethi**  
B.E. Computer Science & Business Systems — Thapar Institute of Engineering & Technology  
 [GitHub](https://github.com/manyasethi777-sketch)
