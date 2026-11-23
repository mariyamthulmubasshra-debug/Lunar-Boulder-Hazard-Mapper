
import os
import re
import math
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        import streamlit as st
        API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        API_KEY = None

client = genai.Client(api_key=API_KEY) if API_KEY else None


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def load_knowledge_base():
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=#{2,3} )", content)

        for section in sections:
            section = section.strip()

            if len(section) > 80:
                source_urls = re.findall(
                    r"(?:Source URL:|URL:)\s*(https?://\S+)",
                    section
                )

                documents.append({
                    "text": section,
                    "source": file_path.name,
                    "urls": source_urls
                })

    if not documents:
        raise ValueError("No knowledge documents found.")

    return documents


def retrieve_relevant_documents(question, documents, top_k=5):
    question_words = set(tokenize(question))
    scored_documents = []

    for document in documents:
        words = tokenize(document["text"])
        word_counts = Counter(words)

        score = sum(
            math.log1p(word_counts[word])
            for word in question_words
        )

        if score > 0:
            scored_documents.append({
                **document,
                "similarity": score
            })

    scored_documents.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return scored_documents[:top_k]


def generate_answer(question, retrieved_documents):
    if not retrieved_documents:
        return "I could not find relevant information in the knowledge base."

    context = "\n\n".join(
        document["text"] for document in retrieved_documents
    )

    prompt = f"""
You are the Lunar Hazard Knowledge Assistant.

Answer using ONLY the provided knowledge context.

Rules:
1. Do not invent scientific facts.
2. If the context does not contain the answer, say so.
3. Explain clearly in simple scientific language.
4. Distinguish project-specific results from established NASA facts.
5. Never claim that a zone is guaranteed safe for landing.

KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

Provide a clear, helpful answer.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        return response.text or "Gemini could not generate an answer."

    except Exception:
        fallback_sections = []

        for document in retrieved_documents:
            lines = document["text"].splitlines()

            if lines and any(
                marker in lines[0].lower()
                for marker in ["references", "verified scientific references"]
            ):
                continue

            paragraphs = []

            for line in lines:
                line = line.strip()

                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("Source URL:")
                    or line.startswith("URL:")
                    or line.startswith("Relevance:")
                    or re.match(r"^\d+\.", line)
                ):
                    continue

                paragraphs.append(line)

            if paragraphs:
                fallback_sections.append(" ".join(paragraphs))

        return (
            "Gemini is temporarily unavailable. "
            "Here is relevant information from the project knowledge base:\n\n"
            + "\n\n".join(fallback_sections[:3])
        )



def ask_lunar_assistant(question):
    documents = load_knowledge_base()

    question_lower = question.lower()

    if any(word in question_lower for word in [
        "metric", "precision", "recall", "map50",
        "evaluation", "test result", "accuracy"
    ]):
        results_file = ROOT / "knowledge" / "project_results.txt"
        content = results_file.read_text(encoding="utf-8")

        if "bounding box" in content.lower() and "segmentation" in content.lower():
            answer = (
                "Independent test evaluation on 260 images "
                "containing 7,162 annotated instances:\n\n"
                "**Bounding-box results**\n"
                "- Precision: 0.666\n"
                "- Recall: 0.600\n"
                "- mAP50: 0.654\n"
                "- mAP50-95: 0.257\n\n"
                "**Segmentation results**\n"
                "- Mask Precision: 0.404\n"
                "- Mask Recall: 0.376\n"
                "- Mask mAP50: 0.293\n"
                "- Mask mAP50-95: 0.073\n\n"
                "These are project-specific test results, not "
                "a guarantee of landing safety."
            )
            return answer, ["project_results.txt"]

    retrieved_documents = retrieve_relevant_documents(
        question,
        documents,
        top_k=5
    )

    answer = generate_answer(question, retrieved_documents)

    sources = []
    seen = set()

    for document in retrieved_documents:
        for url in document.get("urls", []):
            if url not in seen:
                title = document["text"].splitlines()[0].lstrip("# ").strip()

                sources.append({
                    "title": title,
                    "url": url
                })

                seen.add(url)

    if not sources and retrieved_documents:
        sources.append({
            "title": retrieved_documents[0]["source"],
            "url": None
        })

    return answer, sources

