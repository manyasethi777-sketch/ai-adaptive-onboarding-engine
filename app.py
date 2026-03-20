from flask import Flask, render_template, request

app = Flask(__name__)

# -----------------------------
# SKILLS DATABASE
# -----------------------------
skills_db = [
    "python", "c++", "java", "kotlin", "javascript", "html", "css",
    "sql", "mysql", "mongodb", "data structures", "algorithms",
    "machine learning", "deep learning", "nlp", "pandas", "numpy",
    "tensorflow", "pytorch", "flask", "django", "react", "jetpack compose",
    "android development", "git", "github", "oop", "api", "firebase"
]

# -----------------------------
# LEARNING ORDER (priority)
# Lower number = learn first
# -----------------------------
learning_order = {
    "python": 1,
    "c++": 1,
    "java": 1,
    "kotlin": 1,
    "html": 1,
    "css": 1,
    "javascript": 2,
    "oop": 2,
    "git": 2,
    "github": 2,
    "sql": 3,
    "mysql": 3,
    "mongodb": 3,
    "data structures": 4,
    "algorithms": 4,
    "flask": 5,
    "django": 5,
    "react": 5,
    "android development": 5,
    "jetpack compose": 5,
    "api": 5,
    "firebase": 5,
    "numpy": 6,
    "pandas": 6,
    "machine learning": 7,
    "nlp": 8,
    "deep learning": 9,
    "tensorflow": 10,
    "pytorch": 10
}

# -----------------------------
# RESOURCE MAP (for roadmap explanation)
# -----------------------------
resource_map = {
    "python": "Start with Python basics, syntax, loops, functions, and mini practice problems.",
    "c++": "Learn C++ syntax, OOP, STL, and problem solving basics.",
    "java": "Learn Java fundamentals, OOP, collections, and exception handling.",
    "kotlin": "Learn Kotlin syntax, null safety, functions, classes, and Android basics.",
    "html": "Learn page structure, forms, tables, and semantic tags.",
    "css": "Learn styling, layouts, flexbox, grid, and responsive design.",
    "javascript": "Learn variables, DOM, events, functions, arrays, and ES6 features.",
    "sql": "Learn SELECT, WHERE, JOIN, GROUP BY, and practice queries.",
    "mysql": "Understand MySQL basics, schema design, CRUD operations, and joins.",
    "mongodb": "Learn NoSQL basics, documents, collections, and CRUD queries.",
    "data structures": "Study arrays, linked lists, stacks, queues, trees, and graphs.",
    "algorithms": "Practice searching, sorting, recursion, greedy, DP, and graph algorithms.",
    "flask": "Build small backend apps with routes, templates, forms, and APIs.",
    "django": "Learn MVC pattern, models, views, templates, and authentication.",
    "react": "Learn components, props, state, hooks, and build a mini frontend project.",
    "android development": "Understand Android app lifecycle, UI components, and app architecture.",
    "jetpack compose": "Learn composables, state management, layouts, and navigation.",
    "numpy": "Learn arrays, vectorized operations, indexing, and basic math functions.",
    "pandas": "Learn DataFrames, filtering, grouping, cleaning, and CSV handling.",
    "machine learning": "Start with supervised learning, regression, classification, and model evaluation.",
    "deep learning": "Learn neural networks, backpropagation, CNNs, and basic architectures.",
    "nlp": "Learn text preprocessing, tokenization, embeddings, and text classification.",
    "tensorflow": "Build neural network models, train, validate, and save them.",
    "pytorch": "Learn tensors, datasets, training loops, and deep learning workflows.",
    "git": "Learn init, add, commit, push, branch, and version control workflow.",
    "github": "Understand repositories, pull requests, collaboration, and deployment basics.",
    "oop": "Learn classes, objects, inheritance, polymorphism, and encapsulation.",
    "api": "Understand REST APIs, GET/POST requests, JSON, and API integration.",
    "firebase": "Learn authentication, Firestore basics, and backend support for apps."
}

# -----------------------------
# Extract skills from text
# -----------------------------
def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in skills_db:
        if skill.lower() in text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))

# -----------------------------
# Generate roadmap
# -----------------------------
def generate_roadmap(missing_skills):
    # Sort based on learning order
    sorted_skills = sorted(
        missing_skills,
        key=lambda skill: learning_order.get(skill, 999)
    )

    roadmap = []
    for idx, skill in enumerate(sorted_skills, start=1):
        roadmap.append({
            "step": idx,
            "skill": skill,
            "description": resource_map.get(skill, f"Learn the basics and build a mini project in {skill}.")
        })

    return roadmap

# -----------------------------
# Skill match percentage
# -----------------------------
def calculate_match(job_skills, resume_skills):
    if len(job_skills) == 0:
        return 100

    matched = len(job_skills.intersection(resume_skills))
    return round((matched / len(job_skills)) * 100, 2)

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Analyze Route
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = request.form["resume"]
    job_text = request.form["job"]

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    missing_skills = list(job_skills - resume_skills)
    matched_skills = list(job_skills.intersection(resume_skills))

    roadmap = generate_roadmap(missing_skills)
    match_percentage = calculate_match(job_skills, resume_skills)

    return render_template(
        "result.html",
        resume_skills=sorted(list(resume_skills)),
        job_skills=sorted(list(job_skills)),
        matched_skills=sorted(list(matched_skills)),
        missing_skills=sorted(list(missing_skills), key=lambda x: learning_order.get(x, 999)),
        roadmap=roadmap,
        match_percentage=match_percentage
    )

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)