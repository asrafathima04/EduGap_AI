# ============================================================
# EduGap AI — Final Stable Version (gemini-3.6-flash)
# ============================================================

import os
import datetime
import json
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ---- Load environment variables -------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- Page configuration ---------------------------------------------------
st.set_page_config(
    page_title="EduGap AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS -----------------------------------------------------------
st.markdown("""
    <style>
    .main-title { font-size: 2.6rem; font-weight: 800; color: #1f4e79; margin-bottom: 0.2rem; letter-spacing: -1px; }
    .subtitle { color: #5a6b7b; font-size: 1.1rem; margin-bottom: 2rem; }
    
    .concept-card { 
        background: #ffffff; 
        border-left: 4px solid #1f4e79; 
        padding: 1rem; 
        border-radius: 8px; 
        margin-bottom: 0.8rem; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.05); 
        color: #1f2937 !important; 
        font-weight: 500;
    }
    
    .disclaimer { background: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 0.8rem 1.2rem; font-size: 0.9rem; color: #6d4c00; }
    
    .metric-card { 
        background: linear-gradient(135deg, #f0f4f9 0%, #ffffff 100%); 
        border-radius: 10px; 
        padding: 1.2rem; 
        text-align: center; 
        border: 1px solid #e0e7ef; 
        color: #1f2937 !important; 
    }
    .metric-card h3 { color: #1f4e79 !important; margin: 0; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---- API Key check --------------------------------------------------------
if not GEMINI_API_KEY:
    st.error("🔑 **GEMINI_API_KEY not found.** Please set `GEMINI_API_KEY=your_key` in your `.env` file.")
    st.stop()

# ---- Imports --------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
except ImportError:
    st.error("Missing dependencies. Run: `pip install google-genai pydantic`")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# Core Functions
# ============================================================

class ConceptList(BaseModel):
    concepts: list[str]

class GapAnalysis(BaseModel):
    covered: list[str]
    partial: list[str]
    missing: list[str]

def extract_expected_concepts(question: str) -> list[str]:
    prompt = f"List 4 to 8 key concept phrases for this question: {question}"
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=ConceptList.model_json_schema(),
                ),
            )
            data = json.loads(response.text)
            return [c.strip() for c in data.get("concepts", []) if c.strip()][:8]
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep(1)
                continue
            st.warning(f"⚠️ Concept extraction failed: {str(e)}")
            return []
    return []

def analyze_gaps(expected_concepts: list[str], student_answer: str) -> dict:
    if not expected_concepts or not student_answer.strip():
        return {"covered": [], "partial": [], "missing": expected_concepts, "coverage": 0.0}

    prompt = f"""Analyze this student's answer against the expected concepts.
Concepts: {expected_concepts}
Answer: {student_answer}
Return which are covered, partially covered, or missing."""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=GapAnalysis.model_json_schema(),
                ),
            )
            data = json.loads(response.text)
            
            covered = data.get("covered", [])
            partial = data.get("partial", [])
            missing = data.get("missing", [])
            
            total = len(expected_concepts)
            coverage = round(((len(covered) + (len(partial) * 0.5)) / total) * 100, 1) if total > 0 else 0.0

            return {"covered": covered, "partial": partial, "missing": missing, "coverage": coverage}
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep(1)
                continue
            st.error(f"Failed to analyze gaps: {str(e)}")
            return {"covered": [], "partial": [], "missing": expected_concepts, "coverage": 0.0}

def generate_recommendations(question: str, missing: list[str], partial: list[str]) -> str:
    if not missing and not partial:
        return "🎉 Excellent! You covered all expected concepts. Try a harder follow-up question."
    
    gaps_text = ", ".join(missing + partial)
    prompt = f"Student answered: '{question}'. Missed concepts: {gaps_text}. Give 2-3 brief revision suggestions."
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            return response.text.strip()
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep(1)
                continue
            return "📖 Review the missing concepts listed above. Try explaining each one aloud, then rewrite your answer and resubmit."

# ============================================================
# Session State
# ============================================================
if "attempts" not in st.session_state: st.session_state.attempts = []
if "last_result" not in st.session_state: st.session_state.last_result = None

# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=60)
    st.title("EduGap AI")
    st.caption("NLP Knowledge Gap Detection & Adaptive Learning")
    st.divider()

    page = st.radio("Navigation", ["📝 Analyze", "📊 Dashboard", "ℹ️ About"], label_visibility="collapsed")
    st.divider()

    if page == "📝 Analyze":
        st.info("Enter any academic question. The system will extract key concepts and compare them to your answer.")
    elif page == "📊 Dashboard":
        st.info("View your submission history and coverage trends.")

# ============================================================
# Page 1: Analyze
# ============================================================
if page == "📝 Analyze":
    st.markdown('<div class="main-title">🎓 EduGap AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explain concepts in your own words. AI helps you find knowledge gaps.</div>', unsafe_allow_html=True)

    col_q, col_a = st.columns([1, 1])
    with col_q:
        st.subheader("1️⃣ Your Question")
        question = st.text_input("Question", placeholder="e.g., What is data visualization?", label_visibility="collapsed")
    with col_a:
        st.subheader("2️⃣ Your Answer")
        answer = st.text_area("Answer", height=180, placeholder="Write your explanation here...", label_visibility="collapsed")

    if st.button("🔍 Analyze My Answer", type="primary", use_container_width=True):
        if not question.strip() or not answer.strip():
            st.error("Please provide both a question and an answer.")
        elif len(answer.split()) < 5:
            st.error("Your answer is too short. Please write at least a few sentences.")
        else:
            with st.spinner("Extracting key concepts..."):
                concepts = extract_expected_concepts(question)
            if not concepts:
                st.error("Could not extract concepts. Try rephrasing your question.")
            else:
                with st.spinner("Comparing your answer..."):
                    gaps = analyze_gaps(concepts, answer)
                with st.spinner("Generating personalized recommendations..."):
                    rec = generate_recommendations(question, gaps["missing"], gaps["partial"])

                st.session_state.last_result = {
                    "question": question, "answer": answer,
                    "concepts": concepts, "gaps": gaps, "recommendations": rec,
                }
                st.session_state.attempts.append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "question": question, "coverage": gaps["coverage"],
                    "covered": len(gaps["covered"]), "partial": len(gaps["partial"]),
                    "missing": len(gaps["missing"]),
                })

    # Results Display
    if st.session_state.last_result:
        res = st.session_state.last_result
        gaps = res["gaps"]
        st.divider()
        st.subheader("📊 Analysis Results")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><h3>{gaps["coverage"]}%</h3><p>Estimated Coverage</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h3>{len(gaps["covered"])}</h3><p>✅ Covered</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><h3>{len(gaps["missing"])}</h3><p>❌ Missing</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚠️ <b>Note:</b> This percentage is an estimate based on semantic similarity, not a factual correctness score.</div>', unsafe_allow_html=True)

        st.markdown("#### 🧩 Concept Checklist")
        for c in res["concepts"]:
            st.markdown(f'<div class="concept-card">• {c}</div>', unsafe_allow_html=True)

        st.markdown("#### 🔎 Gap Breakdown")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.success("✅ Covered")
            for c in gaps["covered"]: st.write(f"- {c}")
        with g2:
            st.warning("⚠️ Partially Explained")
            for c in gaps["partial"]: st.write(f"- {c}")
        with g3:
            st.error("❌ Missing")
            for c in gaps["missing"]: st.write(f"- {c}")

        st.markdown("#### 📚 Personalized Recommendations")
        st.info(res["recommendations"])

# ============================================================
# Page 2: Dashboard
# ============================================================
elif page == "📊 Dashboard":
    st.markdown('<div class="main-title">📊 Progress Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Track your historical performance and coverage trends.</div>', unsafe_allow_html=True)

    if not st.session_state.attempts:
        st.info("No submissions yet. Go to 'Analyze' and submit an answer first!")
    else:
        df = pd.DataFrame(st.session_state.attempts)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Attempts", len(df))
        m2.metric("Average Coverage", f"{round(df['coverage'].mean(), 1)}%")
        m3.metric("Best Coverage", f"{df['coverage'].max()}%")

        st.markdown("#### Coverage Trend")
        st.line_chart(df.set_index("timestamp")["coverage"])

        st.markdown("#### History")
        st.dataframe(
            df[["timestamp", "question", "coverage", "covered", "partial", "missing"]],
            use_container_width=True, hide_index=True
        )

# ============================================================
# Page 3: About
# ============================================================
else:
    st.markdown('<div class="main-title">ℹ️ About EduGap AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **EduGap AI** is an NLP-based knowledge gap detection tool designed for student self-learning.

    **How it works:**
    1. You enter any academic question. Gemini automatically extracts the key concepts a complete answer should contain.
    2. You explain the concepts in your own words.
    3. The system uses the **Gemini AI** to evaluate which concepts you covered, partially covered, or missed.
    4. For missing or partially explained concepts, Gemini generates personalized revision suggestions.

    **Tech Stack:** Python, Streamlit, Google Gemini API

    **Important Disclaimer:** This tool estimates based on semantic similarity and cannot replace actual factual correctness judgment. Coverage is for reference only; please use it alongside your own understanding.
    """)