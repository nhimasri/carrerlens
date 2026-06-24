# ============================================
# CareerLens — Streamlit Web App
# app.py
# By Himasri | Virtusa Campus Drive 2026
# ============================================

import streamlit as st
import matplotlib.pyplot as plt
from model import run_careerlens, skill_map, importance_df, f1_knn, f1_rf

st.set_page_config(
    page_title="CareerLens",
    page_icon="🎯",
    layout="wide"
)

# ============================================
# Header
# ============================================
st.title("🎯 CareerLens")
st.subheader("AI-Powered Career Intelligence System")
st.markdown("*Answer a few questions and discover your ideal career path — powered by ML*")
st.divider()

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.header("👤 About You")
    name = st.text_input("Your Name", placeholder="e.g. Himasri")

    st.header("🛠️ Your Current Skills")
    skills_input = st.multiselect(
        "Select skills you already have:",
        ['Python', 'SQL', 'R', 'Power BI', 'Statistics',
         'Machine Learning', 'Excel', 'Communication',
         'Figma', 'Docker', 'Linux', 'JavaScript',
         'Deep Learning', 'NLP', 'Tableau'],
        default=['Python', 'SQL']
    )

    st.divider()
    st.markdown("### 🤖 Model Stats")
    st.metric("KNN F1 Score", f"{f1_knn:.2f}")
    st.metric("Random Forest F1", f"{f1_rf:.2f}")
    st.caption("Random Forest selected as best model")

# ============================================
# Personality Assessment
# ============================================
st.header("🧠 Personality Assessment (Holland Code)")
st.markdown("Rate yourself from **0 (not at all)** to **10 (strongly yes)**")

col1, col2, col3 = st.columns(3)
with col1:
    realistic     = st.slider("🔧 Realistic — hands-on, practical", 0, 10, 5)
    investigative = st.slider("🔬 Investigative — analytical, curious", 0, 10, 5)
with col2:
    artistic      = st.slider("🎨 Artistic — creative, expressive", 0, 10, 5)
    social        = st.slider("🤝 Social — helpful, empathetic", 0, 10, 5)
with col3:
    enterprising  = st.slider("💼 Enterprising — leadership, persuasive", 0, 10, 5)
    conventional  = st.slider("📊 Conventional — organized, detail-oriented", 0, 10, 5)

st.divider()

# ============================================
# Aptitude Assessment
# ============================================
st.header("📐 Aptitude Assessment")
col4, col5, col6 = st.columns(3)
with col4:
    logical   = st.slider("🧮 Logical Reasoning", 40, 100, 70)
with col5:
    verbal    = st.slider("💬 Verbal Ability", 40, 100, 70)
with col6:
    numerical = st.slider("🔢 Numerical Ability", 40, 100, 70)

st.divider()

# ============================================
# Work Style
# ============================================
st.header("⚙️ Work Style Preferences")
col7, col8 = st.columns(2)
with col7:
    work_style = st.radio("I prefer working:", ["Solo / Independent", "In a Team"], horizontal=True)
with col8:
    creativity = st.radio("My creativity level:", ["Low / Structured", "High / Open-ended"], horizontal=True)

st.divider()

# ============================================
# Run CareerLens
# ============================================
if st.button("🚀 Discover My Career Path", use_container_width=True, type="primary"):
    if not name:
        st.warning("⚠️ Please enter your name in the sidebar!")
    else:
        scores = {
            'realistic':     realistic,
            'investigative': investigative,
            'artistic':      artistic,
            'social':        social,
            'enterprising':  enterprising,
            'conventional':  conventional,
            'logical':       logical,
            'verbal':        verbal,
            'numerical':     numerical,
            'work_style':    1 if work_style == "In a Team" else 0,
            'creativity':    1 if creativity == "High / Open-ended" else 0
        }

        with st.spinner("🔍 Analysing your profile with AI..."):
            recommendations, top_career, missing_must, missing_good, path = run_careerlens(
                name, skills_input, scores
            )

        st.success(f"✅ CareerLens Report Ready for {name}!")

        # ── Top 3 Career Matches ──
        st.header("🎯 Your Top Career Matches")
        col9, col10, col11 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        colors_hex = ["#2ecc71", "#3498db", "#9b59b6"]

        for i, (_, row) in enumerate(recommendations.iterrows()):
            with [col9, col10, col11][i]:
                st.markdown(f"### {medals[i]} {row['Career']}")
                st.metric("Match Confidence", f"{round(row['Similarity']*100, 1)}%")

        # ── Confidence Chart ──
        st.divider()
        st.header("📊 Confidence Breakdown")
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(
            recommendations['Career'],
            recommendations['Similarity'] * 100,
            color=colors_hex[:len(recommendations)],
            edgecolor='white', linewidth=1.5
        )
        for bar, val in zip(bars, recommendations['Similarity'] * 100):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
        ax.set_ylabel("Match Confidence (%)")
        ax.set_ylim(0, 115)
        ax.set_title(f"CareerLens Report — {name}", fontsize=13, fontweight='bold')
        plt.xticks(rotation=10, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # ── Feature Importance ──
        st.divider()
        st.header("🔍 What Drove Your Recommendation?")
        st.markdown("*These are the top features our Random Forest model used to match you*")
        top_features = importance_df.head(5)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(top_features['Feature'], top_features['Importance'], color='#3498db')
        ax2.set_xlabel("Importance Score")
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)

        # ── Skill Gap Report ──
        st.divider()
        st.header(f"📋 Skill Gap Report — {top_career}")
        col12, col13 = st.columns(2)

        with col12:
            st.error("🚨 Skills You Need ASAP")
            if missing_must:
                for s in missing_must:
                    st.markdown(f"→ **{s}**")
            else:
                st.success("You already have all must-have skills! 🎉")

        with col13:
            st.info("💡 Nice to Have")
            if missing_good:
                for s in missing_good:
                    st.markdown(f"→ {s}")
            else:
                st.success("You have all good-to-have skills too! 🔥")

        st.divider()
        st.success(f"🗺️ Recommended Learning Path: **{path}**")
        st.caption("CareerLens uses Holland Code personality theory + KNN + Random Forest + Cosine Similarity")
