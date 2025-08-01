# Smart Resume Screener

This project is a Streamlit web application that semantically compares a candidate's resume (PDF) with a job description to evaluate how well they match. It uses state-of-the-art NLP models to go beyond simple keyword matching, providing a more meaningful similarity score and actionable feedback.

## Features

- **Upload Resume:** Upload your resume as a PDF file.
- **Paste Job Description:** Paste the job description text into the provided area.
- **Semantic Similarity:** Uses the `all-MiniLM-L6-v2` model from [Sentence Transformers](https://www.sbert.net/) to compute semantic similarity between the resume and job description.
- **Match Score:** Returns a percentage score indicating how well your resume matches the job description.
- **Feedback:** Provides tailored feedback based on the match score.
- **Explanation:** Offers a brief explanation of how the scoring works.

## How It Works

1. **Text Extraction:** The app extracts text from the uploaded PDF resume using the `pypdf` library.
2. **Embedding Generation:** Both the job description and resume text are converted into embeddings using the Sentence Transformer model.
3. **Similarity Calculation:** The cosine similarity between the two embeddings is calculated to produce a match score.
4. **Feedback:** Based on the score, the app provides feedback to help improve your resume alignment.

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

Run the app with:

```sh
streamlit run app.py
```

Then open the provided local URL in your browser.

## File Overview

- [`app.py`](app.py): Main Streamlit application file containing all logic for file upload, text extraction, similarity computation, and UI.
- [`requirements.txt`](requirements.txt): List of required Python packages.

## Notes

- Only PDF resumes are supported.
- The app requires an internet connection to download the pre-trained model on first run.

## License

This project is for educational purpose