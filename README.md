# Fake-News-Detector
9th sem IR Package

PROBLEM STATEMENT :
The spread of online misinformation is a significant societal problem. Manual verification is too slow for the pace of new content, while simple blacklisting is an insufficient solution for evolving claims.
OBJECTIVE :
To design, build, and deploy an automated API server that can receive any news article URL, extract its primary claim in real-time, and deliver an evidence-based verdict on its authenticity by dynamically retrieving and analyzing web-based evidence.
ABSTRACT :
This project implements a full-stack, automated verification pipeline, integrating a Chrome browser extension (frontend) with a Python Flask server (backend).
1.	Browser Extension: The user-facing extension, while on any news article, sends the current tab's URL to the backend API at the click of a button.
2.	API Endpoint: A Flask server provides an API endpoint that receives the URL from the extension.
3.	Information Extraction: The system scrapes the article's text and uses TextRank, a graph-based Extractive Summarization algorithm, to identify the most central and representative sentence to use as the claim.
4.	Evidence Retrieval: This claim is fed into the SerpAPI (Google Search) to retrieve the top 10 most relevant evidence snippets and their source URLs from the live web.
5.	ML Verdict: The core innovation is a Weighted Ensemble Model that combines scores from three independent expert models to form a robust, final verdict:
○	Semantic Model: Uses Spacy’s word embeddings to calculate a semantic score based on the contextual similarity between the claim and the evidence.
○	Sentiment Model: Uses NLTK Vader to calculate a sentiment score. A strong negative score acts as a powerful signal for debunking articles.
○	Source Model: A Rule-Based Classifier calculates a source score by looking up the evidence domains in a static trust model.
A final, weighted average of these three scores (60% semantic, 30% sentiment, 10% source) is calculated. This final score is then mapped to the definitive verdict: "VERIFIED (TRUE)", "VERIFIED (FALSE)", or "UNVERIFIED", which is sent back to the browser extension.
DATASET USED:
The system builds a real-time, on-demand dataset for each request. It retrieves the top 10 organic Google search results (snippets + URLs) using SerpAPI. The analysis is performed directly on these snippets, which serve as high-relevance summaries. This avoids the noise and errors from scraping full, complex web pages, leading to a faster and more precise analysis.
METRICS:
The system uses a weighted ensemble of three core metrics to generate its final qualitative result:
●	semantic_score: (Normalized 0.0-1.0) The average semantic similarity between the claim and the evidence snippets, calculated using spaCy’s word embeddings.
●	sentiment_score: (Normalized -1.0 to 1.0) The average sentiment of the evidence. This is used to detect the negative tone of "debunking" articles.
●	source_score: (Normalized -1.0 to 1.0) The average trust score of the evidence domains, based on the classifier.


