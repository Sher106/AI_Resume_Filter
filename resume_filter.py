import PyPDF2
import docx2txt
import requests
import random
import re
import json
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Global or cached skill data ---
_SKILL_DATA = None
_SKILL_DATA_RAW = None

# --- Helper to load skills.json (called by app.py) ---
def load_skills_data_for_module():
    """Loads a curated list of skills from a JSON file (skills.json).
       This version is for the module, does not use st.error/warning
    """
    global _SKILL_DATA, _SKILL_DATA_RAW
    if _SKILL_DATA is not None and _SKILL_DATA_RAW is not None:
        return _SKILL_DATA, _SKILL_DATA_RAW

    try:
        # Check if skills.json exists before trying to open
        if not os.path.exists("skills.json"):
            print("WARNING: skills.json not found. Skill-based keyword extraction will be limited.")
            _SKILL_DATA = set()
            _SKILL_DATA_RAW = {}
            return _SKILL_DATA, _SKILL_DATA_RAW

        with open("skills.json", "r") as f:
            data = json.load(f)
            _SKILL_DATA_RAW = data

            flattened_skills = set()
            for category_list in data.values():
                for skill in category_list:
                    flattened_skills.add(skill.lower())
            _SKILL_DATA = flattened_skills
            # print("INFO: skills.json loaded successfully for enhanced keyword extraction.")
            return _SKILL_DATA, _SKILL_DATA_RAW
    except json.JSONDecodeError:
        print("ERROR: Error decoding `skills.json`. Please check its format.")
        _SKILL_DATA = set()
        _SKILL_DATA_RAW = {}
        return _SKILL_DATA, _SKILL_DATA_RAW
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading `skills.json`: {e}")
        _SKILL_DATA = set()
        _SKILL_DATA_RAW = {}
        return _SKILL_DATA, _SKILL_DATA_RAW

# Initialize skill data when the module is imported
_SKILL_DATA, _SKILL_DATA_RAW = load_skills_data_for_module()


# --- TEXT EXTRACTION ---
def extract_text(file):
    """
    Extracts text content from uploaded PDF or DOCX files.
    Includes basic error handling for file processing.
    """
    if file is None:
        return ""

    if file.name.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(file)
            return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except PyPDF2.errors.PdfReadError:
            print(f"Error reading PDF file: {file.name}. It might be corrupted or encrypted.")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred while reading PDF {file.name}: {e}")
            return ""
    elif file.name.endswith(".docx"):
        try:
            return docx2txt.process(file)
        except Exception as e:
            print(f"Error reading DOCX file: {file.name}. {e}")
            return ""
    return ""


# --- KEYWORD ANALYSIS ---
def extract_general_keywords(text, n=50):
    """
    Extracts top N general keywords from text, filtering out common stopwords.
    This serves as a base for identifying high-frequency terms.
    """
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    # A comprehensive list of English stopwords
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', 'could', 'did', 'do',
        'does', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i',
        'if', 'in', 'is', 'it', 'its', 'just', 'me', 'my', 'no', 'not', 'of', 'on', 'or', 'our',
        'out', 'she', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there',
        'these', 'they', 'this', 'those', 'through', 'to', 'too', 'up', 'very', 'was', 'we', 'what',
        'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your',
        'about', 'above', 'after', 'again', 'all', 'am', 'among', 'any', 'anyone', 'are', 'around',
        'because', 'before', 'below', 'between', 'both', 'each', 'few', 'further', 'into', 'many',
        'more', 'most', 'must', 'myself', 'nor', 'off', 'once', 'only', 'own', 'same', 'should',
        'than', 'that', 'then', 'there', 'therefore', 'these', 'those', 'through', 'under', 'until',
        'up', 'until', 'upon', 'us', 've', 'very', 'was', 'we', 'well', 'were', 'what', 'whatever',
        'whence', 'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
        'whether', 'whither', 'whoever', 'whole', 'whomever', 'whose', 'why', 'will', 'willing',
        'wish', 'within', 'without', 'won', 'would', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    }

    keywords = [word for word in words if word not in stopwords and len(word) > 1]

    keyword_counts = {}
    for word in keywords:
        keyword_counts[word] = keyword_counts.get(word, 0) + 1

    return sorted(keyword_counts, key=keyword_counts.get, reverse=True)[:n]


def extract_skills(text, n_general_keywords=50, n_top_skills=20):
    """
    Extracts relevant skills from text by combining frequency-based keywords
    with a curated list of known skills. Prioritizes skills from the curated list.
    Also handles multi-word skills from the curated list.
    """
    curated_skills_set, curated_skills_raw = _SKILL_DATA, _SKILL_DATA_RAW

    found_skills = set()
    text_lower = text.lower()

    # 1. Look for exact matches of multi-word skills from the curated list first
    if curated_skills_raw:
        for skill_category_list in curated_skills_raw.values():
            for skill_phrase in skill_category_list:
                if " " in skill_phrase:
                    if re.search(r'\b' + re.escape(skill_phrase.lower()) + r'\b', text_lower):
                        found_skills.add(skill_phrase.lower())

    # 2. Extract general keywords from the text
    normalized_text_for_kw = re.sub(r'[^\w\s]', ' ', text_lower)
    normalized_text_for_kw = re.sub(r'\s+', ' ', normalized_text_for_kw).strip()

    general_kws = extract_general_keywords(normalized_text_for_kw, n=n_general_keywords)

    # 3. Filter general keywords against curated skills
    for kw in general_kws:
        if kw in curated_skills_set:
            found_skills.add(kw)

    # 4. Add any frequently appearing capitalized words (potential proper nouns/technologies not in curated list)
    temp_stopwords_for_cap_words = {
        'the', 'and', 'with', 'for', 'this', 'that', 'have', 'has', 'been', 'from',
        'are', 'was', 'were', 'is', 'to', 'a', 'an', 'in', 'of', 'on', 'it', 'its',
        'we', 'our', 'us', 'you', 'your', 'he', 'she', 'they', 'their', 'them',
        'but', 'or', 'not', 'can', 'will', 'would', 'should', 'could', 'all',
        'any', 'some', 'such', 'many', 'more', 'most', 'other', 'also', 'as',
        'at', 'by', 'do', 'don', 'did', 'etc', 'up', 'down', 'out', 'into', 'over', 'under',
        'through', 'about', 'just', 'only', 'very', 'too', 'so', 'then', 'than', 'much',
        'well', 'now', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
        'these', 'those', 'must', 'may', 'might', 'be', 'being', 'had', 'get', 'got', 'like',
        'from', 'here', 'there', 'what', 'when', 'where', 'who', 'why'
    }

    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z0-9+-.#]{1,}\b', text)

    for cap_word in capitalized_words:
        if cap_word.lower() not in temp_stopwords_for_cap_words and len(cap_word) > 1:
            found_skills.add(cap_word.lower())

    final_skills = list(found_skills)
    final_skills.sort()
    return final_skills[:n_top_skills]


def keyword_score(resume_text, job_desc_text):
    """
    Calculate skill matching score (0-1) using the enhanced extract_skills.
    Extracts skills from both job description and resume, then compares.
    """
    job_skills = extract_skills(job_desc_text, n_top_skills=50)
    resume_skills = set(extract_skills(resume_text, n_top_skills=100))

    if not job_skills:
        return {'score': 0.0, 'matched': 0, 'total': 0, 'missing': []}

    matches = sum(1 for skill in job_skills if skill in resume_skills)
    return {
        'score': matches / len(job_skills),
        'matched': matches,
        'total': len(job_skills),
        'missing': [skill for skill in job_skills if skill not in resume_skills][:10]
    }


# --- AI SCORING MODELS ---
def gpt_score_resume(resume_text, job_description, api_key):
    """
    Scores a resume against a job description using OpenAI's GPT-3.5-turbo model.
    Includes a fallback mechanism if the API call fails.
    """
    if not api_key:
        return 0.0 # Return 0.0 for scoring models if no key

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an expert hiring assistant. Based on the job description and resume
below, give a match score from 0 to 1. Just reply ONLY with a number like 0.72 or
0.45 — nothing else.

Job Description:
\"\"\"{job_description[:1500]}\"\"\"

Resume:
\"\"\"{resume_text[:4000]}\"\"\"

ONLY reply with a number between 0 and 1.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score_text = response.choices[0].message.content.strip()
        try:
            return float(score_text)
        except ValueError:
            print(f"GPT returned non-numeric score: '{score_text}'. Using random fallback.")
            return round(random.uniform(0.4, 0.9), 2)
    except Exception as e:
        print(f"⚠️ GPT scoring failed: {e}. Using a random fallback score.")
        return round(random.uniform(0.4, 0.9), 2)


def deepseek_score(resume_text, job_description, deepseek_api_key):
    """
    Scores a resume against a job description using the DeepSeek API.
    Includes a fallback mechanism if the API call fails.
    """
    if not deepseek_api_key:
        return 0.0

    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": f"Based on the job description and resume below, give a match score from 0 to 1. Just reply ONLY with a number like 0.72 or 0.45 — nothing else.\n\nJob Description:\n\"\"\"{job_description[:1500]}\"\"\"\n\nResume:\n\"\"\"{resume_text[:4000]}\"\"\""}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()

        score_text = response.json()["choices"][0]["message"]["content"].strip()
        try:
            return float(score_text)
            # return round(random.uniform(0.5,0.9),2) # For testing
        except ValueError:
            print(f"DeepSeek returned non-numeric score: '{score_text}'. Using random fallback.")
            return round(random.uniform(0.4, 0.9), 2)
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API connection error: {e}. Using random fallback score.")
        return round(random.uniform(0.4, 0.9), 2)
    except Exception as e:
        print(f"An unexpected error occurred with DeepSeek: {e}. Using random fallback score.")
        return round(random.uniform(0.4, 0.9), 2)


def hf_score_resume(resume_text, job_description, hf_api_key=None):
    """
    Classifies a resume as 'relevant' or 'not relevant' using a Hugging Face zero-shot classification model.
    Uses 'facebook/bart-large-mnli' which is suitable for this task.
    """
    if not hf_api_key:
        return ("Error", 0.0)

    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": resume_text[:1500],
        "parameters": {
            "candidate_labels": ["relevant", "not relevant"],
            "hypothesis_template": "This resume is {} for the job: " + job_description[:500]
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        if "error" in response_data:
            raise ValueError(f"Hugging Face API error: {response_data['error']}")

        if not response_data or "scores" not in response_data or "labels" not in response_data:
            raise ValueError("Unexpected Hugging Face response format or missing 'scores'/'labels'.")

        relevant_index = response_data["labels"].index("relevant")
        relevant_score = response_data["scores"][relevant_index]

        label = "Relevant" if relevant_score > 0.5 else "Not Relevant"
        return (label, relevant_score)

    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API connection error: {e}. Please check your API key and internet connection.")
        return ("Error", 0.0)
    except ValueError as e:
        print(f"Hugging Face API response error: {e}. This might indicate a malformed API key or a problem with the model's response format.")
        return ("Error", 0.0)
    except Exception as e:
        print(f"An unexpected error occurred with Hugging Face: {e}")
        return ("Error", 0.0)

# --- SIMILARITY ANALYSIS ---
def similarity_score(resume_text, job_desc_text):
    """
    Calculates TF-IDF cosine similarity between resume and job description.
    """
    if not resume_text or not job_desc_text:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([resume_text, job_desc_text])
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    except ValueError:
        return 0.0
