"""
=============================================================================
MODULE 05: NATURAL LANGUAGE PROCESSING - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of NLP including:
- 4W+H Explanations (What, Why, When, Where, How)
- Text Preprocessing Techniques
- Sentiment Analysis
- Named Entity Recognition
- Text Classification
- Word Embeddings
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install nltk spacy transformers torch

For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import re
import os
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Try to import optional libraries
try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# GENAI INTEGRATION FOR NLP
# =============================================================================

class NLPAssistant:
    """
    Use GenAI models for NLP tasks
    
    Supports: OpenAI, Anthropic Claude, Google Gemini, Ollama (local), HuggingFace
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def analyze_with_openai(self, text: str, task: str = "sentiment") -> str:
        """Analyze text using OpenAI GPT"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY for AI analysis]"
            
            prompts = {
                "sentiment": f"Analyze the sentiment of this text and explain why: '{text}'",
                "entities": f"Extract all named entities (people, places, organizations, dates) from: '{text}'",
                "summary": f"Summarize this text in one sentence: '{text}'",
                "keywords": f"Extract the 5 most important keywords from: '{text}'",
                "translate": f"Translate this to Spanish: '{text}'"
            }
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompts.get(task, prompts["sentiment"])}],
                max_tokens=300
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install openai: pip install openai]"
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def analyze_with_claude(self, text: str, task: str = "sentiment") -> str:
        """Analyze text using Anthropic Claude"""
        try:
            import anthropic
            if not self.anthropic_key:
                return "[Set ANTHROPIC_API_KEY for Claude analysis]"
            
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Perform {task} analysis on this text: '{text}'"
                }]
            )
            return response.content[0].text
        except ImportError:
            return "[Install anthropic: pip install anthropic]"
        except Exception as e:
            return f"[Claude Error: {e}]"
    
    def analyze_with_ollama(self, text: str, task: str = "sentiment", model: str = "llama2") -> str:
        """Analyze text using Ollama (FREE, local)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Perform {task} analysis on: '{text}'",
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[Ollama error: {response.status_code}]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"
    
    def analyze_with_huggingface(self, text: str, task: str = "sentiment") -> str:
        """Analyze text using HuggingFace Transformers (FREE, local)"""
        if not HAS_TRANSFORMERS:
            return "[Install transformers: pip install transformers torch]"
        
        try:
            if task == "sentiment":
                classifier = pipeline("sentiment-analysis")
                result = classifier(text[:512])  # Limit length
                return f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.2f})"
            elif task == "entities":
                ner = pipeline("ner", aggregation_strategy="simple")
                entities = ner(text[:512])
                return str(entities)
            elif task == "summary":
                summarizer = pipeline("summarization")
                result = summarizer(text[:1024], max_length=100, min_length=30)
                return result[0]['summary_text']
            else:
                return f"[Task '{task}' not supported locally]"
        except Exception as e:
            return f"[HuggingFace Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE NLP EXPLANATION (4W+H)
# =============================================================================

def explain_nlp_comprehensive():
    """Comprehensive 360-degree explanation of NLP"""
    print("=" * 70)
    print("NATURAL LANGUAGE PROCESSING - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS NLP?
============

Natural Language Processing (NLP) is a field of AI that enables computers
to understand, interpret, and generate human language.

KEY COMPONENTS:
    1. Natural Language Understanding (NLU) - Comprehending text
    2. Natural Language Generation (NLG) - Creating text
    3. Speech Recognition - Converting speech to text
    4. Speech Synthesis - Converting text to speech

NLP PIPELINE:
    Raw Text -> Tokenization -> Preprocessing -> Feature Extraction -> Model -> Output

WHY IS NLP IMPORTANT?
=====================

1. MASSIVE DATA: 80% of business data is unstructured text
2. HUMAN INTERFACE: Natural way to interact with computers
3. AUTOMATION: Automate reading, writing, translation
4. INSIGHTS: Extract knowledge from documents

WHEN TO USE NLP?
================

USE NLP FOR:
    - Chatbots and virtual assistants
    - Sentiment analysis (reviews, social media)
    - Document classification
    - Machine translation
    - Search engines
    - Spam detection
    - Question answering

WHERE IS NLP USED?
==================

INDUSTRIES:
    - Healthcare: Medical record analysis, drug discovery
    - Finance: News sentiment, fraud detection
    - Legal: Contract analysis, case research
    - E-commerce: Product reviews, recommendations
    - Customer Service: Chatbots, ticket routing

PRODUCTS:
    - Google Search, Google Translate
    - Siri, Alexa, Google Assistant
    - ChatGPT, Claude, Gemini
    - Grammarly, DeepL

HOW DOES NLP WORK?
==================

TRADITIONAL APPROACH:
    1. Tokenization: Split text into words/sentences
    2. Normalization: Lowercase, remove punctuation
    3. Stopword Removal: Remove common words
    4. Stemming/Lemmatization: Reduce to root form
    5. Feature Extraction: Bag of Words, TF-IDF
    6. Model: Naive Bayes, SVM, etc.

MODERN APPROACH (Deep Learning):
    1. Tokenization: Subword tokenization (BPE, WordPiece)
    2. Embedding: Convert tokens to vectors
    3. Model: Transformer (BERT, GPT)
    4. Fine-tuning: Adapt to specific task
    """)


# =============================================================================
# SECTION 2: HISTORY AND EVOLUTION
# =============================================================================

def nlp_history():
    """History and evolution of NLP"""
    print("\n" + "=" * 70)
    print("HISTORY AND EVOLUTION OF NLP")
    print("=" * 70)
    
    timeline = [
        ("1950s", "Rule-Based Systems", "Hand-crafted grammar rules"),
        ("1960s", "ELIZA", "First chatbot (pattern matching)"),
        ("1980s", "Statistical NLP", "Probabilistic models, HMMs"),
        ("1990s", "Machine Learning", "Naive Bayes, SVM for text"),
        ("2003", "Word2Vec Precursors", "Neural language models"),
        ("2013", "Word2Vec", "Word embeddings revolution"),
        ("2014", "Seq2Seq", "Encoder-decoder for translation"),
        ("2017", "Transformer", "Attention is all you need"),
        ("2018", "BERT", "Bidirectional pre-training"),
        ("2019", "GPT-2", "Large-scale language models"),
        ("2020", "GPT-3", "175B parameters, few-shot learning"),
        ("2022", "ChatGPT", "Conversational AI mainstream"),
        ("2023", "GPT-4, Claude 2", "Multimodal, reasoning"),
    ]
    
    print("\nTIMELINE:")
    print("-" * 70)
    for year, event, description in timeline:
        print(f"{year}: {event} - {description}")


# =============================================================================
# SECTION 3: TEXT PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """Comprehensive text preprocessing toolkit"""
    
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both',
        'either', 'neither', 'not', 'only', 'own', 'same', 'than', 'too',
        'very', 'just', 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your',
        'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'
    }
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split text into words (tokens)"""
        return re.findall(r'\b\w+\b', text.lower())
    
    @staticmethod
    def sentence_tokenize(text: str) -> List[str]:
        """Split text into sentences"""
        return re.split(r'(?<=[.!?])\s+', text)
    
    @classmethod
    def remove_stopwords(cls, tokens: List[str]) -> List[str]:
        """Remove common words that don't carry meaning"""
        return [t for t in tokens if t not in cls.STOPWORDS]
    
    @staticmethod
    def stem(word: str) -> str:
        """Simple Porter-like stemming"""
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment', 'ness', 'tion', 'able']
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    @staticmethod
    def lemmatize(word: str) -> str:
        """Simple lemmatization (irregular verbs)"""
        lemmas = {
            'running': 'run', 'ran': 'run', 'runs': 'run',
            'better': 'good', 'best': 'good',
            'went': 'go', 'goes': 'go', 'going': 'go',
            'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
            'had': 'have', 'has': 'have', 'having': 'have',
        }
        return lemmas.get(word.lower(), word.lower())
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text"""
        return re.sub(r'[^\w\s]', '', text)
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace"""
        return ' '.join(text.split())


def preprocessing_demo():
    """Demo text preprocessing"""
    print("\n" + "=" * 70)
    print("TEXT PREPROCESSING DEMO")
    print("=" * 70)
    
    text = "I'm running to the store! The weather is GREAT today. It's 25 degrees."
    
    print(f"\nOriginal: {text}")
    
    preprocessor = TextPreprocessor()
    
    tokens = preprocessor.tokenize(text)
    print(f"Tokenized: {tokens}")
    
    no_stopwords = preprocessor.remove_stopwords(tokens)
    print(f"Without stopwords: {no_stopwords}")
    
    stemmed = [preprocessor.stem(t) for t in no_stopwords]
    print(f"Stemmed: {stemmed}")
    
    sentences = preprocessor.sentence_tokenize(text)
    print(f"Sentences: {sentences}")


# =============================================================================
# SECTION 4: SENTIMENT ANALYSIS
# =============================================================================

class SentimentAnalyzer:
    """Multiple approaches to sentiment analysis"""
    
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'happy', 'joy', 'best', 'awesome', 'beautiful', 'perfect',
        'brilliant', 'outstanding', 'superb', 'delightful', 'pleasant',
        'positive', 'nice', 'fine', 'enjoy', 'like', 'recommend'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor',
        'disappointing', 'sad', 'angry', 'ugly', 'boring', 'stupid',
        'annoying', 'disgusting', 'pathetic', 'negative', 'dislike',
        'fail', 'failed', 'wrong', 'problem', 'issue', 'broken'
    }
    
    INTENSIFIERS = {'very', 'really', 'extremely', 'absolutely', 'totally'}
    NEGATORS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing'}
    
    @classmethod
    def rule_based(cls, text: str) -> Tuple[str, float]:
        """Rule-based sentiment analysis"""
        tokens = TextPreprocessor.tokenize(text)
        
        pos_score = 0
        neg_score = 0
        multiplier = 1
        negate = False
        
        for i, token in enumerate(tokens):
            if token in cls.NEGATORS:
                negate = True
                continue
            if token in cls.INTENSIFIERS:
                multiplier = 2
                continue
            
            if token in cls.POSITIVE_WORDS:
                if negate:
                    neg_score += multiplier
                else:
                    pos_score += multiplier
            elif token in cls.NEGATIVE_WORDS:
                if negate:
                    pos_score += multiplier
                else:
                    neg_score += multiplier
            
            multiplier = 1
            negate = False
        
        total = pos_score + neg_score
        if total == 0:
            return "Neutral", 0.5
        
        score = pos_score / total
        if score > 0.6:
            return "Positive", score
        elif score < 0.4:
            return "Negative", 1 - score
        else:
            return "Neutral", 0.5
    
    @staticmethod
    def vader_style(text: str) -> Dict[str, float]:
        """VADER-style sentiment (simplified)"""
        tokens = TextPreprocessor.tokenize(text)
        
        # Simplified VADER lexicon
        lexicon = {
            'love': 3.2, 'hate': -3.4, 'good': 1.9, 'bad': -2.5,
            'great': 3.1, 'terrible': -3.2, 'amazing': 3.1, 'awful': -3.0,
            'happy': 2.7, 'sad': -2.1, 'excellent': 3.2, 'poor': -2.5,
        }
        
        scores = [lexicon.get(t, 0) for t in tokens]
        compound = sum(scores) / (math.sqrt(sum(s**2 for s in scores) + 15) if scores else 1)
        
        return {
            'compound': compound,
            'positive': sum(1 for s in scores if s > 0) / len(tokens) if tokens else 0,
            'negative': sum(1 for s in scores if s < 0) / len(tokens) if tokens else 0,
            'neutral': sum(1 for s in scores if s == 0) / len(tokens) if tokens else 1
        }


def sentiment_demo():
    """Demo sentiment analysis"""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS DEMO")
    print("=" * 70)
    
    texts = [
        "I love this product! It's absolutely amazing!",
        "This is terrible. I hate it. Worst purchase ever.",
        "The product is okay. Nothing special.",
        "Not bad, but not great either.",
        "I really don't like this at all."
    ]
    
    analyzer = SentimentAnalyzer()
    
    for text in texts:
        sentiment, score = analyzer.rule_based(text)
        print(f"\nText: '{text}'")
        print(f"Sentiment: {sentiment} (confidence: {score:.2f})")


# =============================================================================
# SECTION 5: NAMED ENTITY RECOGNITION
# =============================================================================

class EntityExtractor:
    """Extract named entities from text"""
    
    @staticmethod
    def extract_patterns(text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns"""
        return {
            'emails': re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text),
            'urls': re.findall(r'https?://\S+', text),
            'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
            'phone': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
            'money': re.findall(r'\$[\d,]+(?:\.\d{2})?', text),
            'percentages': re.findall(r'\d+(?:\.\d+)?%', text),
            'capitalized': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        }
    
    @staticmethod
    def extract_with_context(text: str, entity_type: str) -> List[Tuple[str, str]]:
        """Extract entities with surrounding context"""
        patterns = {
            'person': r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            'organization': r'(?:Inc\.|Corp\.|LLC|Ltd\.|Company)\b',
            'location': r'(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        }
        
        pattern = patterns.get(entity_type, r'\b[A-Z][a-z]+\b')
        matches = re.finditer(pattern, text)
        
        results = []
        for match in matches:
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            results.append((match.group(), context))
        
        return results


def ner_demo():
    """Demo named entity recognition"""
    print("\n" + "=" * 70)
    print("NAMED ENTITY RECOGNITION DEMO")
    print("=" * 70)
    
    text = """
    Contact John Smith at john.smith@example.com or call 555-123-4567.
    Visit our website at https://example.com for more information.
    The meeting is scheduled for 12/25/2024. The project costs $50,000
    and is expected to increase revenue by 25%.
    """
    
    print(f"Text: {text}")
    
    extractor = EntityExtractor()
    entities = extractor.extract_patterns(text)
    
    print("\nExtracted Entities:")
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {values}")


# =============================================================================
# SECTION 6: TEXT SIMILARITY AND EMBEDDINGS
# =============================================================================

class TextSimilarity:
    """Calculate text similarity using various methods"""
    
    @staticmethod
    def jaccard(text1: str, text2: str) -> float:
        """Jaccard similarity (set overlap)"""
        set1 = set(TextPreprocessor.tokenize(text1))
        set2 = set(TextPreprocessor.tokenize(text2))
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def cosine(vec1: List[float], vec2: List[float]) -> float:
        """Cosine similarity between vectors"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    @staticmethod
    def bag_of_words(texts: List[str]) -> Tuple[List[str], List[List[int]]]:
        """Convert texts to bag of words vectors"""
        all_words = set()
        for text in texts:
            all_words.update(TextPreprocessor.tokenize(text))
        
        vocab = sorted(all_words)
        vectors = []
        
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            counts = Counter(tokens)
            vector = [counts.get(word, 0) for word in vocab]
            vectors.append(vector)
        
        return vocab, vectors
    
    @staticmethod
    def tfidf(texts: List[str]) -> Tuple[List[str], List[List[float]]]:
        """Calculate TF-IDF vectors"""
        vocab, bow_vectors = TextSimilarity.bag_of_words(texts)
        
        # Calculate document frequency
        df = [sum(1 for vec in bow_vectors if vec[i] > 0) for i in range(len(vocab))]
        n_docs = len(texts)
        
        # Calculate TF-IDF
        tfidf_vectors = []
        for vec in bow_vectors:
            total_terms = sum(vec)
            tfidf_vec = []
            for i, count in enumerate(vec):
                tf = count / total_terms if total_terms > 0 else 0
                idf = math.log(n_docs / (df[i] + 1)) + 1
                tfidf_vec.append(tf * idf)
            tfidf_vectors.append(tfidf_vec)
        
        return vocab, tfidf_vectors


def similarity_demo():
    """Demo text similarity"""
    print("\n" + "=" * 70)
    print("TEXT SIMILARITY DEMO")
    print("=" * 70)
    
    texts = [
        "The cat sat on the mat",
        "The dog sat on the rug",
        "Machine learning is fascinating",
    ]
    
    sim = TextSimilarity()
    
    print("\nJaccard Similarity:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = sim.jaccard(texts[i], texts[j])
            print(f"  '{texts[i][:30]}...' vs '{texts[j][:30]}...': {score:.3f}")
    
    print("\nTF-IDF Cosine Similarity:")
    vocab, tfidf_vecs = sim.tfidf(texts)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = sim.cosine(tfidf_vecs[i], tfidf_vecs[j])
            print(f"  Text {i+1} vs Text {j+1}: {score:.3f}")


# =============================================================================
# SECTION 7: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common NLP interview questions"""
    print("\n" + "=" * 70)
    print("NLP INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is the difference between stemming and lemmatization?",
         "Stemming cuts word endings (running->runn), lemmatization uses vocabulary (running->run). Lemmatization is more accurate but slower."),
        ("What is TF-IDF?",
         "Term Frequency-Inverse Document Frequency. Measures word importance: high TF (appears often in doc) + high IDF (rare across docs) = important word."),
        ("Explain word embeddings.",
         "Dense vector representations of words where similar words have similar vectors. Examples: Word2Vec, GloVe, FastText."),
        ("What is attention mechanism?",
         "Allows model to focus on relevant parts of input. Self-attention compares each word to all others. Key component of Transformers."),
        ("What is BERT?",
         "Bidirectional Encoder Representations from Transformers. Pre-trained on masked language modeling. Used for classification, NER, QA."),
        ("How do you handle out-of-vocabulary words?",
         "Subword tokenization (BPE, WordPiece), character-level models, or UNK token with context."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 8: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common mistakes in NLP"""
    print("\n" + "=" * 70)
    print("COMMON NLP PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Not handling text encoding", "Always use UTF-8, handle special characters"),
        ("Ignoring class imbalance", "Use stratified sampling, class weights, or oversampling"),
        ("Data leakage in preprocessing", "Fit tokenizer/vectorizer only on training data"),
        ("Not handling negation", "'not good' is negative, not positive"),
        ("Ignoring context", "Word meaning depends on context (bank: river vs financial)"),
        ("Over-relying on accuracy", "Use F1, precision, recall for imbalanced data"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 05: NLP - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = NLPAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is NLP? (4W+H)")
        print("  2. History and Evolution")
        print()
        print("TECHNIQUES:")
        print("  3. Text Preprocessing Demo")
        print("  4. Sentiment Analysis Demo")
        print("  5. Named Entity Recognition Demo")
        print("  6. Text Similarity Demo")
        print()
        print("DEEP DIVE:")
        print("  7. Interview Questions")
        print("  8. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  9. AI Sentiment Analysis (OpenAI)")
        print("  10. AI Sentiment Analysis (HuggingFace - FREE)")
        print("  11. AI Sentiment Analysis (Ollama - FREE)")
        print()
        print("  12. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-12): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_nlp_comprehensive()
        elif choice == "2":
            nlp_history()
        elif choice == "3":
            preprocessing_demo()
        elif choice == "4":
            sentiment_demo()
        elif choice == "5":
            ner_demo()
        elif choice == "6":
            similarity_demo()
        elif choice == "7":
            interview_questions()
        elif choice == "8":
            common_pitfalls()
        elif choice == "9":
            text = input("Enter text to analyze: ").strip()
            print(assistant.analyze_with_openai(text, "sentiment"))
        elif choice == "10":
            text = input("Enter text to analyze: ").strip()
            print(assistant.analyze_with_huggingface(text, "sentiment"))
        elif choice == "11":
            text = input("Enter text to analyze: ").strip()
            print(assistant.analyze_with_ollama(text, "sentiment"))
        elif choice == "12":
            explain_nlp_comprehensive()
            nlp_history()
            preprocessing_demo()
            sentiment_demo()
            ner_demo()
            similarity_demo()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
