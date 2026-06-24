# ============================================
# CareerLens — AI Career Intelligence System
# model.py — ML Core
# By Himasri | Virtusa Campus Drive 2026
# ============================================

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1 — Generate Synthetic Dataset
# ============================================
np.random.seed(42)
n = 500

data = {
    'realistic':      np.random.randint(0, 11, n),
    'investigative':  np.random.randint(0, 11, n),
    'artistic':       np.random.randint(0, 11, n),
    'social':         np.random.randint(0, 11, n),
    'enterprising':   np.random.randint(0, 11, n),
    'conventional':   np.random.randint(0, 11, n),
    'logical':        np.random.randint(40, 101, n),
    'verbal':         np.random.randint(40, 101, n),
    'numerical':      np.random.randint(40, 101, n),
    'work_style':     np.random.randint(0, 2, n),
    'creativity':     np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

def assign_career(row):
    if row['investigative'] >= 7 and row['logical'] >= 70:
        return 'Data Science / AI'
    elif row['realistic'] >= 7 and row['numerical'] >= 70:
        return 'Engineering / DevOps'
    elif row['artistic'] >= 7 and row['creativity'] == 1:
        return 'UX / Product Design'
    elif row['social'] >= 7 and row['verbal'] >= 70:
        return 'HR / Education / Counselling'
    elif row['enterprising'] >= 7 and row['verbal'] >= 65:
        return 'Business / Management'
    elif row['conventional'] >= 7 and row['numerical'] >= 65:
        return 'Finance / Analytics'
    else:
        return 'General Technology'

df['career_cluster'] = df.apply(assign_career, axis=1)

# ============================================
# STEP 2 — Train Models
# ============================================
X = df.drop('career_cluster', axis=1)
y = df['career_cluster']
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# KNN (baseline)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
f1_knn = f1_score(y_test, knn.predict(X_test), average='weighted')

# Random Forest (optimized)
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), params,
                    cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
f1_rf = f1_score(y_test, best_model.predict(X_test), average='weighted')

# Feature Importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

# ============================================
# STEP 3 — Skill Map
# ============================================
skill_map = {
    'Data Science / AI': {
        'must_have': ['Python', 'Machine Learning', 'Statistics', 'SQL'],
        'good_to_have': ['Deep Learning', 'NLP', 'Cloud (AWS/GCP)', 'Tableau'],
        'learning_path': 'Kaggle courses → Scikit-learn → TensorFlow → MLOps'
    },
    'Engineering / DevOps': {
        'must_have': ['Linux', 'Docker', 'CI/CD', 'Python/Bash'],
        'good_to_have': ['Kubernetes', 'Terraform', 'AWS', 'Monitoring tools'],
        'learning_path': 'Linux basics → Docker → Jenkins → Kubernetes'
    },
    'UX / Product Design': {
        'must_have': ['Figma', 'User Research', 'Wireframing', 'Prototyping'],
        'good_to_have': ['HTML/CSS', 'Motion Design', 'Design Systems'],
        'learning_path': 'Figma basics → UX Research → Case Studies → Portfolio'
    },
    'HR / Education / Counselling': {
        'must_have': ['Communication', 'Empathy', 'Excel', 'HRMS tools'],
        'good_to_have': ['Power BI', 'People Analytics', 'Python basics'],
        'learning_path': 'HR fundamentals → People Analytics → SHRM certification'
    },
    'Business / Management': {
        'must_have': ['Excel', 'Communication', 'Problem Solving', 'PowerPoint'],
        'good_to_have': ['SQL', 'Power BI', 'Project Management', 'CRM tools'],
        'learning_path': 'Business Analytics → MBA prep → PMP certification'
    },
    'Finance / Analytics': {
        'must_have': ['Excel', 'SQL', 'Financial Modeling', 'Statistics'],
        'good_to_have': ['Python', 'Power BI', 'Bloomberg', 'R'],
        'learning_path': 'Financial modeling → SQL → Python for Finance → CFA'
    },
    'General Technology': {
        'must_have': ['Python', 'SQL', 'Problem Solving', 'Git'],
        'good_to_have': ['Cloud basics', 'APIs', 'Agile', 'Linux'],
        'learning_path': 'Python → SQL → Cloud fundamentals → Specialization'
    }
}

# ============================================
# STEP 4 — Recommendation Engine
# ============================================
def get_career_recommendations(user_input: dict):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)

    # Cosine similarity against career cluster centroids
    career_clusters = df.groupby('career_cluster')[feature_names].mean()
    career_scaled = scaler.transform(career_clusters)
    similarities = cosine_similarity(user_scaled, career_scaled)[0]

    career_scores = pd.DataFrame({
        'Career': career_clusters.index,
        'Similarity': similarities
    }).sort_values('Similarity', ascending=False)

    return career_scores.head(3).reset_index(drop=True)


def skill_gap_report(top3_careers, user_current_skills: list):
    top_career = top3_careers.iloc[0]['Career']
    skills = skill_map[top_career]
    missing_must = [s for s in skills['must_have'] if s not in user_current_skills]
    missing_good = [s for s in skills['good_to_have'] if s not in user_current_skills]
    return top_career, missing_must, missing_good, skills['learning_path']


def run_careerlens(name, skills, scores):
    recommendations = get_career_recommendations(scores)
    top_career, missing_must, missing_good, path = skill_gap_report(recommendations, skills)
    return recommendations, top_career, missing_must, missing_good, path
