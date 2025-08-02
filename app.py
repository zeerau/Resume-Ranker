import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pypdf
import torch

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to compute semantic similarity
def compute_similarity(job_desc, resume_text):
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return round(similarity_score * 100, 2)

# Function to give basic feedback
def get_feedback(score):
    if score > 80:
        return "✅ Excellent match! Your resume strongly aligns with the job description."
    elif score > 40:
        return "⚠️ Good match, but you could improve by tailoring your resume to include more relevant experience or terms."
    else:
        return "❌ Weak match. Try aligning your resume better with the job requirements, responsibilities, and tools mentioned."

# Streamlit UI
st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("🧠 Smart Resume Screener")

# Add project overview
st.markdown("""
## 📝 Overview

This app helps you evaluate how well your resume matches a job description using advanced semantic analysis. 
Upload your resume (PDF) and paste the job description to receive a match score and actionable feedback. 
The app uses a state-of-the-art language model to understand context and meaning, not just keywords.
""")

st.markdown("Upload your **resume** and paste the **job description** to see how well they match semantically.")

job_description = st.text_area("📄 Paste Job Description Here", height=250)
uploaded_resume = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])

if st.button("🔍 Evaluate Match"):
    if not job_description or not uploaded_resume:
        st.warning("Please provide both a job description and a resume.")
    else:
        with st.spinner("Analyzing..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_resume)
                score = compute_similarity(job_description, resume_text)
                feedback = get_feedback(score)

                st.success(f"Match Score: **{score}%**")
                st.markdown(feedback)

                with st.expander("🔍 Explanation"):
                    st.markdown(
                        f"""
                        - The model uses **semantic similarity** rather than just keyword matching.
                        - This means it understands context, roles, and meaning.
                        - A high score means your resume aligns well in terms of **skills, experience, and language** used.
                        """
                    )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
