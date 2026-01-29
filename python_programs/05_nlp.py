"""
Module 05: Natural Language Processing (NLP)
============================================
Learn how computers understand human language!
"""

import re
from collections import Counter

# ============================================
# Text Preprocessing
# ============================================

def tokenize(text):
    """Split text into words (tokens)"""
    return re.findall(r'\b\w+\b', text.lower())

def remove_stopwords(tokens):
    """Remove common words that don't carry meaning"""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'as', 'into', 'through', 'during', 'before', 'after', 'and',
                 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
                 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he',
                 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'}
    return [t for t in tokens if t not in stopwords]

def stem(word):
    """Simple stemming - reduce words to root form"""
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

# ============================================
# Sentiment Analysis
# ============================================

def analyze_sentiment(text):
    """Simple rule-based sentiment analysis"""
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                     'fantastic', 'love', 'happy', 'joy', 'best', 'awesome',
                     'beautiful', 'perfect', 'brilliant', 'outstanding'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst',
                     'poor', 'disappointing', 'sad', 'angry', 'ugly', 'boring',
                     'stupid', 'annoying', 'disgusting', 'pathetic'}
    
    tokens = tokenize(text)
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    
    if pos_count > neg_count:
        return "Positive", pos_count - neg_count
    elif neg_count > pos_count:
        return "Negative", neg_count - pos_count
    else:
        return "Neutral", 0

# ============================================
# Named Entity Recognition (Simple)
# ============================================

def extract_entities(text):
    """Extract named entities using patterns"""
    entities = {
        'emails': re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text),
        'urls': re.findall(r'https?://\S+', text),
        'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
        'phone': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
        'capitalized': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    }
    return entities

# ============================================
# Text Similarity
# ============================================

def jaccard_similarity(text1, text2):
    """Calculate similarity between two texts"""
    set1 = set(tokenize(text1))
    set2 = set(tokenize(text2))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# ============================================
# Bag of Words
# ============================================

def bag_of_words(texts):
    """Convert texts to bag of words representation"""
    all_words = set()
    for text in texts:
        all_words.update(tokenize(text))
    
    vocab = sorted(all_words)
    vectors = []
    
    for text in texts:
        tokens = tokenize(text)
        counts = Counter(tokens)
        vector = [counts.get(word, 0) for word in vocab]
        vectors.append(vector)
    
    return vocab, vectors

# ============================================
# Demo
# ============================================

def main():
    print("=" * 60)
    print("Natural Language Processing Demo")
    print("=" * 60)
    
    text = "I love this amazing product! It's the best thing I've ever bought."
    
    print(f"\nOriginal text: {text}")
    print(f"\nTokens: {tokenize(text)}")
    print(f"Without stopwords: {remove_stopwords(tokenize(text))}")
    
    sentiment, score = analyze_sentiment(text)
    print(f"\nSentiment: {sentiment} (score: {score})")
    
    text2 = "Contact us at info@example.com or visit https://example.com"
    print(f"\nEntities in '{text2}':")
    for entity_type, entities in extract_entities(text2).items():
        if entities:
            print(f"  {entity_type}: {entities}")
    
    # Similarity
    t1 = "The cat sat on the mat"
    t2 = "The dog sat on the rug"
    print(f"\nSimilarity between '{t1}' and '{t2}': {jaccard_similarity(t1, t2):.2f}")

if __name__ == "__main__":
    main()
