import os
import json
import re
import logging
from collections import Counter
from urllib.parse import urlparse
import numpy as np

# --- Flask and Server Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Web Scraping and Search ---
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

# --- IR & ML Imports  ---
import spacy
import nltk
from nltk.corpus import stopwords
import pytextrank 

# --- ML/NLP Models ---
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Configuration ---
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = 'screenshots'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- (A) Download NLTK/spaCy Models  ---
try:
    nlp = spacy.load("en_core_web_md")
    # --- Add TextRank to the spaCy pipeline ---
    nlp.add_pipe("textrank")
except IOError:
    print("Downloading spaCy model 'en_core_web_md'...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    print("Adding TextRank to pipeline...")
    nlp.add_pipe("textrank")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading NLTK VADER sentiment lexicon...")
    nltk.download('vader_lexicon')

# --- (B) Initialize IR Tools ---
stop_words = set(stopwords.words('english'))
sentiment_analyzer = SentimentIntensityAnalyzer()


# --- (C) User's Dictionaries (for Source Credibility) ---
SOURCE_CREDIBILITY_MODEL = {
    # High Trust
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "bbc.com": 0.9,
    "pbs.org": 0.9,
    "npr.org": 0.9,
    "factcheck.org": 0.9,
    "snopes.com": 0.8,
    "politifact.com": 0.8,
    "theguardian.com": 0.8,
    "nytimes.com": 0.8,
    "washingtonpost.com": 0.8,
    "wsj.com": 0.8,
    "c-span.org": 0.9,
    "aljazeera.com": 0.7,
    "nature.com": 0.9,
    "science.org": 0.9,
    "scientificamerican.com": 0.8,
    "techcrunch.com": 0.7,
    "arstechnica.com": 0.7,
    "theverge.com": 0.7,
    "wired.com": 0.7,
    "cnet.com": 0.7,
    "abcnews.go.com": 0.8,
    "cbsnews.com": 0.8,
    "nbcnews.com": 0.8,
    "verified-news.com": 0.8,
    "trusted-source.net": 0.8,

    "thehindu.com": 0.8,
    "indianexpress.com": 0.8,
    "livemint.com": 0.8,
    "ndtv.com": 0.7,
    "indiatoday.in": 0.7,
    "timesofindia.indiatimes.com": 0.7,
    "hindustantimes.com": 0.7,
    
    # Low Trust
    "fake-news-site.com": -1.0,
    "misleading-info.org": -1.0,
    "conspiracy-theories.net": -1.0,
    "unverified-claims.com": -0.8,
    "false-statistics.org": -0.8
}


# --- (D) Helper Functions (Web Retrieval) ---

def search_google(query):
    """Searches Google for the given query using SerpAPI."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        logger.error("SERPAPI_API_KEY environment variable not set.")
        return []
        
    params = {"q": query, "api_key": api_key, "num": 10}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        return [{"url": r['link'], "snippet": r.get('snippet', ''), "source": r.get('source', '')} for r in organic_results]
    except Exception as e:
        logger.error(f"Error in SerpAPI search: {e}")
        return []

def extract_text_from_url(url):
    """Extracts all paragraph text from a web page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        
        if not text:
             text = ' '.join(soup.stripped_strings)
             
        return text
    except requests.RequestException as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""

# --- (E)  IR/ML Pipeline Functions ---

# --- MODEL 1: TextRank (Information Extraction) ---
def extract_claim_textrank(text):
    """
    (Information Extraction - TextRank)
    Uses a graph-based IR algorithm to find the most central
    and important sentence to use as the "claim".
    """

    doc = nlp(text[:1000000])
    
    if not doc._.textrank or not doc._.textrank.summary:
        logger.warning("TextRank failed, falling back to simple NER.")

        for sent in doc.sents:
            if sent.ents:
                return sent.text.strip()
        return text.strip().split('\n')[0] 

    top_sentence = list(doc._.textrank.summary(limit_sentences=1))[0]
    return top_sentence.text.strip()


# --- MODEL 2: Semantic Similarity (VSM/Word Embeddings) ---
def get_semantic_similarity_score(claim_doc, evidence_docs_text):
    """
    (Retrieval Models - Semantic Similarity)
    Calculates a single normalized score [0, 1] for semantic similarity.
    """
    evidence_docs = [nlp(text[:1000000]) for text in evidence_docs_text if text and nlp(text[:1000000]).vector_norm]
    
    if not evidence_docs:
        return 0.0

    similarities = np.array([claim_doc.similarity(doc) for doc in evidence_docs])
    avg_sim = float(similarities.mean())
    
    # Normalize the score: A raw score of 0.35 is "bad" (0.0), 0.8 is "good" (1.0)
    min_sim = 0.35
    max_sim = 0.80
    normalized_score = (avg_sim - min_sim) / (max_sim - min_sim)
    
    return float(np.clip(normalized_score, 0.0, 1.0))


# --- MODEL 3: Sentiment Analysis (VADER) ---
def get_sentiment_score(evidence_docs_text):
    """
    (Text Mining - Sentiment Analysis)
    Analyzes sentiment and returns a single score [-1, 1].
    A strong negative score is a high-confidence signal for "FALSE" (debunking).
    """

    sentiments = [sentiment_analyzer.polarity_scores(text[:1000000]) for text in evidence_docs_text if text]
    if not sentiments:
        return 0.0

    compound_scores = [s['compound'] for s in sentiments]
    avg_compound = float(np.mean(compound_scores))

    if avg_compound < -0.1:
        return avg_compound * 2 # Amplify negative
    else:
        return avg_compound # Return positive/neutral as is
        
    return float(np.clip(avg_compound, -1.0, 1.0))


# --- MODEL 4: Source Credibility (Rule-Based Classifier) ---
def get_source_credibility_score(urls):
    """
    (Syllabus: Text Categorization - rule-based)
    Looks up domains in our SOURCE_CREDIBILITY_MODEL.
    """
    score = 0
    count = 0
    for url in urls:
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            if domain in SOURCE_CREDIBILITY_MODEL:
                score += SOURCE_CREDIBILITY_MODEL[domain]
                count += 1
        except Exception:
            continue
    
    return score / count if count > 0 else 0.0

# --- FINAL VERDICT: Weighted Ensemble Model ---
def get_weighted_ensemble_verdict(features):
    """
    (Ensemble Methods - Weighted Voting)
    Combines all our models into a  verdict.
    """
    
    # --- Define Weights for each Model ---
    WEIGHTS = {
        'semantic': 0.6,
        'sentiment': 0.3,
        'source': 0.1
    }
    
    sem_score = features["semantic_score"]
    sent_score = features["sentiment_score"]
    source_score = features["source_score"]

    logger.info(f"Ensemble Model Inputs: [Semantic: {sem_score:.2f}], [Sentiment: {sent_score:.2f}], [Source: {source_score:.2f}]")

    # --- 1. Calculate the final score  ---
    semantic_vote = (sem_score * 2) - 1
    
    sentiment_vote = sent_score
    
    source_vote = source_score
    
    final_score = (
        (semantic_vote * WEIGHTS['semantic']) +
        (sentiment_vote * WEIGHTS['sentiment']) +
        (source_vote * WEIGHTS['source'])
    )
    
    logger.info(f"Final Weighted Score: {final_score:.3f}")

    # --- 2. Convert final score to a verdict ---
    if final_score > 0.4:
        return "VERIFIED (TRUE)"
    elif final_score < -0.3:
        return "VERIFIED (FALSE)"
    elif sem_score < 0.1: # Override: if semantic score was 0, no evidence was found
        return "UNVERIFIED (No Relevant Evidence Found)"
    else:
        return "UNVERIFIED (Conflicting Evidence)"


# --- (F) Main Flask Route ---

@app.route('/api/capture', methods=['POST'])
def capture():
    try:
        if 'screenshot' not in request.files:
            return jsonify({"status": "error", "message": "No screenshot file"}), 400
            
        file = request.files['screenshot']
        url = request.form.get('url', 'no-url')
        
        if url == 'no-url' or not url.startswith('http'):
             return jsonify({"status": "error", "message": "No valid URL provided."}), 400

        # --- 1. Information Extraction (with TextRank) ---
        logger.info(f"Scraping text from URL: {url}")
        raw_text = extract_text_from_url(url)
        if not raw_text:
            return jsonify({"status": "error", "message": f"Could not extract text from {url}"}), 400
            
        claim = extract_claim_textrank(raw_text)
        claim_doc = nlp(claim) # Get spacy doc for claim
        logger.info(f"Extracted Claim: {claim}")

        # --- 2. Retrieval (IR) ---
        logger.info("Retrieving documents from Google...")
        retrieved_results = search_google(claim)
        
        retrieved_urls = [r['url'] for r in retrieved_results]
        retrieved_snippets = [r['snippet'] for r in retrieved_results if r.get('snippet')]
        
        # --- 3. Feature Generation (Run all models) ---

        logger.info("Running Ensemble Models on Evidence...")
        
        features = {
            "semantic_score": get_semantic_similarity_score(claim_doc, retrieved_snippets),
            "sentiment_score": get_sentiment_score(retrieved_snippets),
            "source_score": get_source_credibility_score(retrieved_urls)
        }

        # --- 4. Get Final Verdict (Weighted Voting) ---
        logger.info("Calculating final verdict...")
        final_verdict = get_weighted_ensemble_verdict(features)
        
        logger.info(f"Final Verdict: {final_verdict}")
        
        # --- 5. Final Output ---
        return jsonify({
            "status": "success",
            "message": final_verdict,
            "claim": claim,
            "ir_features": features, # Return all the model scores
            "retrieved_evidence": retrieved_results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /api/capture: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"Internal Server Error: {str(e)}"}), 500


@app.route('/api/misleading', methods=['GET'])
def get_misleading_websites():
    data = {k: v for k, v in SOURCE_CREDIBILITY_MODEL.items() if v < 0}
    return jsonify({ 'status': 'success', 'data': data })

@app.route('/api/correct', methods=['GET'])
def get_correct_websites():
    data = {k: v for k, v in SOURCE_CREDIBILITY_MODEL.items() if v > 0}
    return jsonify({ 'status': 'success', 'data': data })

# --- Main Entry Point ---
if __name__ == '__main__':
    print("\n--- Starting IR/ML Fake News Detection Server (v5 - ENSEMBLE HEURISTICS) ---")
    print("This server uses TextRank, Semantic Similarity, Sentiment Analysis, and a Weighted Voting Ensemble.")
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)