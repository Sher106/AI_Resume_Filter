import streamlit as st
import json
import os
from resume_filter import (
    extract_text,
    extract_skills,
    keyword_score,
    gpt_score_resume,
    deepseek_score,
    hf_score_resume,
    similarity_score
)

# --- Streamlit Application Setup ---
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("📊 Smart Resume Analyzer")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Keys
    openai_key = st.text_input("OpenAI Key", type="password", help="Your OpenAI API Key for GPT-3.5 scoring.")
    deepseek_key = st.text_input("DeepSeek Key", type="password", help="Your DeepSeek API Key for DeepSeek model scoring.")
    hf_api_key = st.text_input("Hugging Face API Key", type="password", help="Your Hugging Face Access Token for Inference API (e.g., for 'facebook/bart-large-mnli').")

    # Job Description Source
    st.subheader("Job Description")
    jd_source = st.radio("Source", ["Upload File", "Paste Text", "Load from job_description.json"])

    job_desc_text = ""
    if jd_source == "Upload File":
        jd_file = st.file_uploader("Upload JD", type=["txt", "pdf", "docx"])
        job_desc_text = extract_text(jd_file) if jd_file else ""
    elif jd_source == "Paste Text":
        job_desc_text = st.text_area("Paste Job Description", height=200)
    elif jd_source == "Load from job_description.json":
        try:
            # Check if job_description.json exists before trying to open
            if not os.path.exists("job_description.json"):
                st.error("❌ `job_description.json` not found. Please ensure it's in the same directory as your script.")
            else:
                with open("job_description.json", "r") as f:
                    job_data_from_json = json.load(f)
                    job_desc_text = f"{job_data_from_json.get('title', 'Job Title not found')}\n" \
                                    f"Skills: {', '.join(job_data_from_json.get('required_skills', []))}\n" \
                                    f"Experience: {job_data_from_json.get('experience', 'N/A')}\n" \
                                    f"Education: {job_data_from_json.get('education', 'N/A')}"
                    st.success("✅ Job description loaded from 'job_description.json'.")
        except json.JSONDecodeError:
            st.error("❌ Error parsing `job_description.json`. Please check its JSON format.")
        except Exception as e:
            st.error(f"❌ Failed to load job description from JSON: {e}")
        st.info("Job description from JSON:\n" + job_desc_text if job_desc_text else "No JD loaded from JSON.")


# --- MAIN INTERFACE ---
st.subheader("📄 Upload Resumes")
uploaded_files = st.file_uploader("📤 Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and job_desc_text:
    st.subheader("🧠 Evaluation Results")

    # Extract skills from job description text once
    skills_from_jd = extract_skills(job_desc_text)
    with st.expander("🔍 Extracted Skills from Job Description"):
        if skills_from_jd:
            st.write(", ".join(skills_from_jd))
        else:
            st.write("No specific skills extracted. Ensure job description has relevant terms.")

    # Process each resume
    for file in uploaded_files:
        st.markdown(f"---")
        st.subheader(f"Analyzing: {file.name}")
        resume_raw_text = extract_text(file)

        if not resume_raw_text:
            st.warning(f"Skipping **{file.name}**: Could not extract text. Please ensure it's a valid PDF or DOCX.")
            continue

        # Create tabs for different analysis methods
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔑 Keyword Match",
            "🤖 AI Scoring",
            "📊 Similarity",
            "Full Analysis"
        ])

        with tab1: # Keyword analysis
            st.write("### Keyword Match Score")
            result = keyword_score(resume_raw_text, job_desc_text)
            st.metric("Match Score", f"{result['score'] * 100:.0f}%")
            st.progress(result['score'])

            col1_kw, col2_kw = st.columns(2)
            with col1_kw:
                st.write(f"✅ **{result['matched']}/{result['total']}** keywords matched")
            with col2_kw:
                if result['missing']:
                    st.warning("Top Missing Keywords:")
                    for kw in result['missing']:
                        st.write(f"- {kw}")
                else:
                    st.info("All targeted keywords found!")


        with tab2: # AI models
            st.write("### AI Model Scores")
            col1_ai, col2_ai, col3_ai = st.columns(3)
            with col1_ai:
                if openai_key:
                    with st.spinner(f"Scoring {file.name} with GPT-3.5..."):
                        score = gpt_score_resume(resume_raw_text, job_desc_text, openai_key)
                    st.metric("GPT-3.5 Score", f"{score:.2f}")
                else:
                    st.info("Enter OpenAI Key to use GPT-3.5.")
            with col2_ai:
                if deepseek_key:
                    with st.spinner(f"Scoring {file.name} with DeepSeek..."):
                        score = deepseek_score(resume_raw_text, job_desc_text, deepseek_key)
                    st.metric("DeepSeek Score", f"{score:.2f}")
                else:
                    st.info("Enter DeepSeek Key to use DeepSeek.")
            with col3_ai:
                if hf_api_key:
                    with st.spinner(f"Scoring {file.name} with Hugging Face..."):
                        label, score = hf_score_resume(resume_raw_text, job_desc_text, hf_api_key)
                    st.metric("HF Zero-Shot Score", f"{score:.2f}")
                    st.caption(f"Prediction: **{label}**")
                else:
                    st.info("Enter Hugging Face Key to use Hugging Face.")


        with tab3: # Similarity
            st.write("### Text Similarity Score")
            score = similarity_score(resume_raw_text, job_desc_text)
            st.metric("TF-IDF Cosine Similarity", f"{score:.2f}")
            st.caption("(TF-IDF cosine similarity measures text content similarity)")

        with tab4: # Combined report
            st.subheader("Comprehensive Analysis")

            # Keyword analysis
            kws_result = keyword_score(resume_raw_text, job_desc_text)
            st.write(f"**Keyword Match:** {kws_result['matched']}/{kws_result['total']} ({kws_result['score'] * 100:.0f}%)")
            if kws_result['missing']:
                st.write("Missing Keywords: " + ", ".join(kws_result['missing']))
            else:
                st.write("All targeted keywords found!")

            # Similarity
            sim_score = similarity_score(resume_raw_text, job_desc_text)
            st.write(f"**Text Similarity:** {sim_score:.2f}")

            # AI scores
            st.write("**AI Predictions:**")
            cols_full_ai = st.columns(3)
            if openai_key:
                with cols_full_ai[0]:
                    score = gpt_score_resume(resume_raw_text, job_desc_text, openai_key)
                    st.metric("GPT-3.5", f"{score:.2f}")
            else:
                with cols_full_ai[0]: st.info("OpenAI Key needed.")
            if deepseek_key:
                with cols_full_ai[1]:
                    score = deepseek_score(resume_raw_text, job_desc_text, deepseek_key)
                    st.metric("DeepSeek", f"{score:.2f}")
            else:
                with cols_full_ai[1]: st.info("DeepSeek Key needed.")
            if hf_api_key:
                with cols_full_ai[2]:
                    label, score = hf_score_resume(resume_raw_text, job_desc_text, hf_api_key)
                    st.metric("HF Zero-Shot", f"{score:.2f}")
                    st.caption(f"({label})")
            else:
                with cols_full_ai[2]: st.info("Hugging Face Key needed.")

else:
    st.info("Please upload resumes and provide a job description to start the analysis.")