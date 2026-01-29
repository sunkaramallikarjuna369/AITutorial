"""
Module 11: RAG (Retrieval-Augmented Generation)
===============================================
Give AI access to your own documents and data!
"""

def explain_rag():
    print("=" * 60)
    print("RAG - Retrieval-Augmented Generation")
    print("=" * 60)
    print("""
RAG combines search with AI generation for accurate, up-to-date answers!

The Problem:
- LLMs have knowledge cutoff dates
- They can hallucinate (make things up)
- They don't know your private data

The Solution: RAG!
1. RETRIEVE: Search your documents for relevant info
2. AUGMENT: Add that info to the prompt
3. GENERATE: Let the LLM answer using the context

How it works:
    User Question
         ↓
    [Vector Search] → Find relevant documents
         ↓
    [Combine] → Question + Retrieved docs
         ↓
    [LLM] → Generate answer with citations

Benefits:
- Always up-to-date (just update documents)
- Grounded in facts (reduces hallucination)
- Works with private data
- Provides sources/citations
    """)

code_examples = '''
"""
RAG Implementation Examples
===========================
"""

# ============================================
# 1. Simple RAG with LangChain
# ============================================

from langchain.document_loaders import TextLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_simple_rag(documents_path):
    # Load documents
    loader = TextLoader(documents_path)
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain


# ============================================
# 2. RAG from Scratch
# ============================================

import numpy as np
from openai import OpenAI

client = OpenAI()

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text, metadata=None):
        """Add a document to the knowledge base"""
        # Get embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        
        self.documents.append({
            "text": text,
            "metadata": metadata or {}
        })
        self.embeddings.append(embedding)
    
    def search(self, query, top_k=3):
        """Find most relevant documents"""
        # Get query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort and return top results
        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                "document": self.documents[idx],
                "score": sim
            })
        
        return results
    
    def query(self, question):
        """Answer a question using RAG"""
        # Retrieve relevant documents
        results = self.search(question)
        
        # Build context
        context = "\\n\\n".join([r["document"]["text"] for r in results])
        
        # Generate answer
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""Answer based on the following context:
                    
{context}

If the answer is not in the context, say "I don't have information about that."
"""
                },
                {"role": "user", "content": question}
            ]
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": results
        }


# ============================================
# 3. RAG with LlamaIndex
# ============================================

from llama_index import VectorStoreIndex, SimpleDirectoryReader

def create_llamaindex_rag(directory):
    # Load documents
    documents = SimpleDirectoryReader(directory).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    return query_engine


# ============================================
# 4. Advanced RAG with Reranking
# ============================================

from sentence_transformers import CrossEncoder

class AdvancedRAG(SimpleRAG):
    def __init__(self):
        super().__init__()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def search_with_rerank(self, query, top_k=3, rerank_top_k=10):
        """Search with reranking for better results"""
        # Initial retrieval
        initial_results = self.search(query, top_k=rerank_top_k)
        
        # Rerank
        pairs = [(query, r["document"]["text"]) for r in initial_results]
        scores = self.reranker.predict(pairs)
        
        # Sort by reranker scores
        reranked = sorted(
            zip(scores, initial_results),
            key=lambda x: x[0],
            reverse=True
        )
        
        return [r for _, r in reranked[:top_k]]


# ============================================
# 5. Hybrid Search (Vector + Keyword)
# ============================================

from rank_bm25 import BM25Okapi

class HybridRAG(SimpleRAG):
    def __init__(self):
        super().__init__()
        self.bm25 = None
        self.tokenized_docs = []
    
    def add_document(self, text, metadata=None):
        super().add_document(text, metadata)
        
        # Add to BM25 index
        tokens = text.lower().split()
        self.tokenized_docs.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def hybrid_search(self, query, top_k=3, alpha=0.5):
        """Combine vector and keyword search"""
        # Vector search
        vector_results = self.search(query, top_k=top_k*2)
        
        # BM25 search
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        
        # Combine scores
        combined = {}
        for i, result in enumerate(vector_results):
            idx = self.documents.index(result["document"])
            vector_score = result["score"]
            bm25_score = bm25_scores[idx]
            
            # Normalize and combine
            combined[idx] = alpha * vector_score + (1-alpha) * bm25_score
        
        # Sort and return
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [self.documents[idx] for idx, _ in sorted_results[:top_k]]


print("RAG examples loaded!")
print("Install: pip install langchain llama-index chromadb sentence-transformers rank-bm25")
'''

def main():
    explain_rag()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
