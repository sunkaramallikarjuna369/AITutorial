"""
=============================================================================
MODULE 11: RAG (RETRIEVAL-AUGMENTED GENERATION) - COMPREHENSIVE 360 DEGREE
=============================================================================

This module provides COMPLETE coverage of RAG including:
- 4W+H Explanations (What, Why, When, Where, How)
- RAG Architecture and Components
- Vector Databases and Embeddings
- Implementation Examples
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install numpy chromadb sentence-transformers

For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import os
import math
from typing import List, Dict, Tuple, Optional

# Try to import optional libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class RAGAssistant:
    """
    Demonstrate RAG with multiple GenAI models
    
    Supports: OpenAI, Anthropic Claude, Google Gemini, Ollama (local)
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def get_embedding_openai(self, text: str) -> List[float]:
        """Get embedding using OpenAI"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return []
            
            client = OpenAI(api_key=self.openai_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except:
            return []
    
    def generate_with_context(self, question: str, context: str, provider: str = "openai") -> str:
        """Generate answer using retrieved context"""
        prompt = f"""Answer the question based ONLY on the following context:

Context:
{context}

Question: {question}

If the answer is not in the context, say "I don't have information about that."

Answer:"""
        
        if provider == "openai":
            return self._call_openai(prompt)
        elif provider == "claude":
            return self._call_claude(prompt)
        elif provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return "[Unknown provider]"
    
    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY]"
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def _call_claude(self, prompt: str) -> str:
        try:
            import anthropic
            if not self.anthropic_key:
                return "[Set ANTHROPIC_API_KEY]"
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"[Claude Error: {e}]"
    
    def _call_ollama(self, prompt: str, model: str = "llama2") -> str:
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[Ollama error: {response.status_code}]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE RAG EXPLANATION (4W+H)
# =============================================================================

def explain_rag_comprehensive():
    """Comprehensive 360-degree explanation of RAG"""
    print("=" * 70)
    print("RAG (RETRIEVAL-AUGMENTED GENERATION) - COMPREHENSIVE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS RAG?
============

RAG (Retrieval-Augmented Generation) combines information retrieval with
text generation to produce accurate, grounded, and up-to-date responses.

THE PROBLEM WITH PURE LLMs:
    1. Knowledge cutoff date (don't know recent events)
    2. Hallucinations (make up facts)
    3. No access to private/proprietary data
    4. Can't cite sources

THE RAG SOLUTION:
    1. RETRIEVE: Search relevant documents from knowledge base
    2. AUGMENT: Add retrieved context to the prompt
    3. GENERATE: LLM answers using the provided context

RAG PIPELINE:
    User Question
         |
         v
    [Embedding Model] -> Query Vector
         |
         v
    [Vector Database] -> Top-K Similar Documents
         |
         v
    [Prompt Construction] -> Question + Context
         |
         v
    [LLM] -> Grounded Answer with Citations

WHY USE RAG?
============

1. ACCURACY: Answers grounded in actual documents
2. UP-TO-DATE: Just update the knowledge base
3. PRIVATE DATA: Works with your proprietary information
4. CITATIONS: Can provide sources for answers
5. COST-EFFECTIVE: Smaller LLM + retrieval often beats larger LLM
6. CONTROLLABLE: Easy to update/remove information

WHEN TO USE RAG?
================

USE RAG FOR:
    - Customer support (product documentation)
    - Legal research (case law, contracts)
    - Medical Q&A (clinical guidelines)
    - Enterprise search (internal documents)
    - Educational assistants (textbooks, courses)

DON'T USE RAG FOR:
    - Creative writing (no factual grounding needed)
    - General conversation
    - Tasks where LLM knowledge is sufficient

WHERE IS RAG USED?
==================

PRODUCTS:
    - Perplexity AI (search + generation)
    - ChatGPT with browsing
    - Microsoft Copilot
    - Enterprise chatbots

FRAMEWORKS:
    - LangChain
    - LlamaIndex
    - Haystack
    - Semantic Kernel

HOW DOES RAG WORK?
==================

STEP 1: INDEXING (Offline)
    Documents -> Chunking -> Embedding -> Vector Database
    
STEP 2: RETRIEVAL (Online)
    Query -> Embedding -> Vector Search -> Top-K Documents
    
STEP 3: GENERATION (Online)
    Prompt = "Context: {documents}\n\nQuestion: {query}\n\nAnswer:"
    Response = LLM(Prompt)
    """)


# =============================================================================
# SECTION 2: RAG COMPONENTS
# =============================================================================

def rag_components():
    """Detailed explanation of RAG components"""
    print("\n" + "=" * 70)
    print("RAG COMPONENTS")
    print("=" * 70)
    
    print("""
1. DOCUMENT LOADERS
   ================
   Load documents from various sources:
   - PDF, Word, PowerPoint
   - Web pages, URLs
   - Databases, APIs
   - Code repositories
   
   Tools: LangChain loaders, Unstructured, PyPDF

2. TEXT SPLITTERS (CHUNKING)
   =========================
   Split documents into smaller chunks:
   
   Strategies:
   - Fixed size (e.g., 500 tokens)
   - Sentence-based
   - Paragraph-based
   - Semantic (by topic)
   
   Key parameters:
   - chunk_size: Size of each chunk
   - chunk_overlap: Overlap between chunks (for context)
   
   Example:
   chunk_size=1000, overlap=200
   -> Chunk 1: tokens 0-1000
   -> Chunk 2: tokens 800-1800
   -> Chunk 3: tokens 1600-2600

3. EMBEDDING MODELS
   =================
   Convert text to dense vectors:
   
   Models:
   - OpenAI: text-embedding-3-small, text-embedding-3-large
   - Cohere: embed-english-v3.0
   - Open-source: sentence-transformers, E5, BGE
   
   Dimensions: 384 to 3072 (higher = more expressive but slower)

4. VECTOR DATABASES
   =================
   Store and search embeddings efficiently:
   
   Options:
   - Pinecone (managed, scalable)
   - Weaviate (open-source, hybrid search)
   - Chroma (lightweight, local)
   - Milvus (open-source, scalable)
   - FAISS (Facebook, in-memory)
   - Qdrant (open-source, Rust-based)
   
   Search types:
   - Approximate Nearest Neighbor (ANN)
   - Exact search (small datasets)
   - Hybrid (vector + keyword)

5. RETRIEVAL STRATEGIES
   =====================
   - Dense retrieval: Vector similarity
   - Sparse retrieval: BM25, TF-IDF
   - Hybrid: Combine dense + sparse
   - Reranking: Use cross-encoder to reorder results
   - Multi-query: Generate multiple queries, combine results

6. GENERATION (LLM)
   =================
   Generate answer using retrieved context:
   
   Prompt template:
   "Use the following context to answer the question.
    If you don't know, say 'I don't know'.
    
    Context: {retrieved_documents}
    
    Question: {user_question}
    
    Answer:"
    """)


# =============================================================================
# SECTION 3: SIMPLE RAG IMPLEMENTATION
# =============================================================================

class SimpleRAG:
    """Simple RAG implementation from scratch"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.assistant = RAGAssistant()
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Simple TF-based embedding (for demo without API)"""
        if not HAS_NUMPY:
            return [0.0] * 100
        
        # Simple word frequency embedding
        words = text.lower().split()
        vocab = list(set(words))[:100]  # Limit vocab size
        
        vector = np.zeros(100)
        for i, word in enumerate(vocab):
            vector[i] = words.count(word) / len(words) if words else 0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def add_document(self, text: str, metadata: Dict = None):
        """Add document to knowledge base"""
        embedding = self._simple_embedding(text)
        self.documents.append({
            "text": text,
            "metadata": metadata or {},
            "embedding": embedding
        })
        print(f"Added document: {text[:50]}...")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if not self.documents:
            return []
        
        query_embedding = self._simple_embedding(query)
        
        # Calculate cosine similarities
        similarities = []
        for i, doc in enumerate(self.documents):
            if HAS_NUMPY:
                sim = np.dot(query_embedding, doc["embedding"])
            else:
                sim = sum(a * b for a, b in zip(query_embedding, doc["embedding"]))
            similarities.append((sim, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top-k
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                "text": self.documents[idx]["text"],
                "score": sim,
                "metadata": self.documents[idx]["metadata"]
            })
        
        return results
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Answer question using RAG"""
        # Retrieve
        results = self.search(question, top_k)
        
        if not results:
            return {"answer": "No documents in knowledge base.", "sources": []}
        
        # Build context
        context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(results)])
        
        # Generate (using simple response for demo)
        answer = f"Based on the retrieved documents:\n\n{context}\n\nThe answer relates to the above context."
        
        return {
            "answer": answer,
            "sources": results
        }


def simple_rag_demo():
    """Demo simple RAG implementation"""
    print("\n" + "=" * 70)
    print("SIMPLE RAG DEMO")
    print("=" * 70)
    
    rag = SimpleRAG()
    
    # Add sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        "Machine learning is a subset of AI that enables systems to learn from data. Common algorithms include linear regression, decision trees, and neural networks.",
        "RAG (Retrieval-Augmented Generation) combines search with AI generation. It retrieves relevant documents and uses them to generate accurate answers.",
        "Transformers are neural network architectures that use self-attention. They power models like GPT, BERT, and T5.",
        "Vector databases store embeddings for fast similarity search. Popular options include Pinecone, Weaviate, and Chroma."
    ]
    
    print("\nAdding documents to knowledge base:")
    for doc in documents:
        rag.add_document(doc)
    
    # Query
    question = "What is RAG?"
    print(f"\nQuestion: {question}")
    
    result = rag.query(question)
    print(f"\nAnswer:\n{result['answer']}")
    
    print("\nSources:")
    for i, source in enumerate(result['sources']):
        print(f"  [{i+1}] Score: {source['score']:.3f}")
        print(f"      {source['text'][:80]}...")


# =============================================================================
# SECTION 4: ADVANCED RAG TECHNIQUES
# =============================================================================

def advanced_rag_techniques():
    """Advanced RAG techniques"""
    print("\n" + "=" * 70)
    print("ADVANCED RAG TECHNIQUES")
    print("=" * 70)
    
    print("""
1. HYBRID SEARCH
   ==============
   Combine vector search with keyword search (BM25):
   
   final_score = alpha * vector_score + (1-alpha) * bm25_score
   
   Benefits: Better for exact matches + semantic understanding

2. RERANKING
   ==========
   Use a cross-encoder to reorder initial results:
   
   1. Retrieve top-100 with fast vector search
   2. Rerank with cross-encoder to get top-10
   
   Models: ms-marco-MiniLM, bge-reranker

3. QUERY EXPANSION
   ================
   Generate multiple queries from user question:
   
   Original: "What is RAG?"
   Expanded: 
   - "What is Retrieval-Augmented Generation?"
   - "How does RAG work?"
   - "RAG architecture and components"
   
   Retrieve for all queries, combine results

4. HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE)
   ========================================
   Generate a hypothetical answer, embed that instead of query:
   
   1. User asks: "What is RAG?"
   2. LLM generates hypothetical answer
   3. Embed the hypothetical answer
   4. Search with that embedding
   
   Better for complex questions

5. CONTEXTUAL COMPRESSION
   =======================
   Compress retrieved documents to relevant parts:
   
   1. Retrieve full documents
   2. Use LLM to extract only relevant sentences
   3. Pass compressed context to generation
   
   Reduces token usage, improves focus

6. PARENT DOCUMENT RETRIEVAL
   =========================
   Store small chunks for retrieval, return larger context:
   
   1. Index: Small chunks (100 tokens)
   2. Retrieve: Find matching small chunks
   3. Return: Parent document (1000 tokens)
   
   Better context while maintaining precision

7. SELF-RAG
   =========
   Model decides when to retrieve and evaluates relevance:
   
   1. Generate initial response
   2. Model asks: "Do I need more information?"
   3. If yes, retrieve and regenerate
   4. Model evaluates: "Is retrieved info relevant?"
    """)


# =============================================================================
# SECTION 5: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common RAG interview questions"""
    print("\n" + "=" * 70)
    print("RAG INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is RAG and why is it useful?",
         "RAG combines retrieval with generation. Useful because: reduces hallucinations, enables access to private data, provides citations, keeps knowledge up-to-date."),
        ("How do you choose chunk size?",
         "Trade-off: Smaller chunks = more precise retrieval but less context. Larger chunks = more context but may include irrelevant info. Typical: 500-1000 tokens with 10-20% overlap."),
        ("What is the difference between dense and sparse retrieval?",
         "Dense: Uses embeddings, captures semantic meaning. Sparse: Uses keyword matching (BM25), exact matches. Hybrid combines both for best results."),
        ("How do you evaluate RAG systems?",
         "Metrics: Retrieval (Recall@K, MRR), Generation (BLEU, ROUGE, human eval), End-to-end (answer accuracy, faithfulness, relevance)."),
        ("How do you handle documents that don't fit in context?",
         "Strategies: Better chunking, summarization, map-reduce (summarize chunks then combine), iterative retrieval."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 6: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common RAG mistakes"""
    print("\n" + "=" * 70)
    print("COMMON RAG PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Poor chunking strategy", "Experiment with chunk size, use semantic chunking, ensure overlap"),
        ("Ignoring metadata", "Use metadata for filtering (date, source, category)"),
        ("Not reranking", "Initial retrieval is fast but imprecise; reranking improves quality"),
        ("Wrong embedding model", "Match embedding model to your domain; fine-tune if needed"),
        ("Stuffing too much context", "More context isn't always better; compress or summarize"),
        ("No evaluation pipeline", "Set up metrics to measure retrieval and generation quality"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 11: RAG - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = RAGAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is RAG? (4W+H)")
        print("  2. RAG Components")
        print()
        print("IMPLEMENTATION:")
        print("  3. Simple RAG Demo")
        print("  4. Advanced RAG Techniques")
        print()
        print("DEEP DIVE:")
        print("  5. Interview Questions")
        print("  6. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  7. RAG with OpenAI")
        print("  8. RAG with Ollama (FREE)")
        print()
        print("  9. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-9): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_rag_comprehensive()
        elif choice == "2":
            rag_components()
        elif choice == "3":
            simple_rag_demo()
        elif choice == "4":
            advanced_rag_techniques()
        elif choice == "5":
            interview_questions()
        elif choice == "6":
            common_pitfalls()
        elif choice == "7":
            context = input("Enter context (knowledge): ").strip()
            question = input("Enter question: ").strip()
            print("\nAnswer:")
            print(assistant.generate_with_context(question, context, "openai"))
        elif choice == "8":
            context = input("Enter context (knowledge): ").strip()
            question = input("Enter question: ").strip()
            print("\nAnswer:")
            print(assistant.generate_with_context(question, context, "ollama"))
        elif choice == "9":
            explain_rag_comprehensive()
            rag_components()
            simple_rag_demo()
            advanced_rag_techniques()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
