import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Database, Search, FileText, Code, Brain } from 'lucide-react'

const Module11RAG = () => {
  const [query, setQuery] = useState('')
  const [retrievedDocs, setRetrievedDocs] = useState<string[]>([])
  const [answer, setAnswer] = useState('')
  const [step, setStep] = useState(0)

  const documents = [
    "Python was created by Guido van Rossum and released in 1991.",
    "JavaScript is the most popular programming language for web development.",
    "Machine learning is a subset of artificial intelligence.",
    "React is a JavaScript library for building user interfaces.",
    "TensorFlow is an open-source machine learning framework by Google."
  ]

  const simulateRAG = () => {
    if (!query.trim()) return
    setStep(1)
    
    setTimeout(() => {
      const relevant = documents.filter(doc => 
        doc.toLowerCase().includes(query.toLowerCase().split(' ')[0])
      )
      setRetrievedDocs(relevant.length > 0 ? relevant : [documents[2]])
      setStep(2)
      
      setTimeout(() => {
        setAnswer(`Based on the retrieved documents: ${relevant.length > 0 ? relevant[0] : 'Machine learning is a fascinating field of AI that enables computers to learn from data.'}`)
        setStep(3)
      }, 1000)
    }, 1000)
  }

  const pythonCode = `# RAG - Retrieval Augmented Generation
# Give LLMs access to external knowledge!

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PDFLoader

# ============================================
# Understanding RAG
# ============================================

"""
RAG = Retrieval + Generation

Problem: LLMs have knowledge cutoff and can hallucinate
Solution: Give them access to your own documents!

How it works:
1. CHUNK: Split documents into smaller pieces
2. EMBED: Convert text to vectors (numbers)
3. STORE: Save vectors in a database
4. RETRIEVE: Find relevant chunks for a query
5. GENERATE: LLM answers using retrieved context
"""

# ============================================
# Step 1: Load and Chunk Documents
# ============================================

def load_documents(file_path):
    """Load documents from various sources"""
    if file_path.endswith('.pdf'):
        loader = PDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def chunk_documents(documents, chunk_size=1000, overlap=200):
    """
    Split documents into smaller chunks
    
    Why chunk?
    - LLMs have context limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't lose context at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    return splitter.split_documents(documents)

# Example
documents = load_documents("knowledge_base.txt")
chunks = chunk_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ============================================
# Step 2: Create Embeddings
# ============================================

def create_embeddings():
    """
    Embeddings convert text to vectors
    Similar meanings = similar vectors!
    """
    # OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Or use free alternatives:
    # from langchain.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return embeddings

# Test embedding
embeddings = create_embeddings()
vector = embeddings.embed_query("What is machine learning?")
print(f"Vector dimension: {len(vector)}")  # Usually 1536 for OpenAI

# ============================================
# Step 3: Create Vector Store
# ============================================

def create_vector_store(chunks, embeddings):
    """
    Store vectors in a database for fast retrieval
    
    Popular options:
    - Chroma: Simple, local
    - FAISS: Fast, by Facebook
    - Pinecone: Cloud-based, scalable
    - Weaviate: Open-source, feature-rich
    """
    # Using Chroma (local)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Or using FAISS
    # vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

vectorstore = create_vector_store(chunks, embeddings)

# ============================================
# Step 4: Retrieval
# ============================================

def retrieve_relevant_docs(vectorstore, query, k=3):
    """
    Find the most relevant documents for a query
    
    k = number of documents to retrieve
    """
    # Simple similarity search
    docs = vectorstore.similarity_search(query, k=k)
    
    # Or with scores
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    return docs

# Example
query = "How does machine learning work?"
relevant_docs = retrieve_relevant_docs(vectorstore, query)
for doc in relevant_docs:
    print(f"Content: {doc.page_content[:100]}...")

# ============================================
# Step 5: Generation with Context
# ============================================

def create_rag_chain(vectorstore):
    """
    Create a chain that retrieves and generates
    """
    llm = OpenAI(temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Put all docs in context
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain

rag_chain = create_rag_chain(vectorstore)

# Ask a question
result = rag_chain({"query": "What is machine learning?"})
print(f"Answer: {result['result']}")
print(f"Sources: {[doc.metadata for doc in result['source_documents']]}")

# ============================================
# Complete RAG Pipeline
# ============================================

class RAGPipeline:
    """
    Complete RAG system from scratch
    """
    
    def __init__(self, documents_path):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        
        # Load and process documents
        docs = load_documents(documents_path)
        chunks = chunk_documents(docs)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            chunks, self.embeddings
        )
    
    def query(self, question, k=3):
        """
        Answer a question using RAG
        """
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(question, k=k)
        
        # Build context
        context = "\\n\\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with context
        prompt = f"""Answer the question based on the following context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        answer = self.llm(prompt)
        
        return {
            "answer": answer,
            "sources": relevant_docs
        }

# Usage
rag = RAGPipeline("my_documents.txt")
result = rag.query("What are the benefits of machine learning?")
print(result["answer"])

# ============================================
# Advanced: Hybrid Search
# ============================================

from langchain.retrievers import BM25Retriever, EnsembleRetriever

def create_hybrid_retriever(documents, embeddings):
    """
    Combine keyword search (BM25) with semantic search
    Best of both worlds!
    """
    # Keyword-based retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    # Semantic retriever
    vectorstore = FAISS.from_documents(documents, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Combine them
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6]  # Weight semantic search higher
    )
    
    return ensemble_retriever

# ============================================
# Advanced: Query Expansion
# ============================================

def expand_query(query, llm):
    """
    Generate multiple versions of a query
    for better retrieval
    """
    prompt = f"""Generate 3 different ways to ask this question:
    
Original: {query}

Variations:
1."""
    
    response = llm(prompt)
    variations = [query] + response.strip().split("\\n")
    return variations

# ============================================
# Using LlamaIndex (Alternative to LangChain)
# ============================================

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is RAG?")
print(response)`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">RAG - Retrieval Augmented Generation</h1>
        <p className="text-xl text-purple-200">Give AI access to your own knowledge!</p>
      </div>

      <Tabs defaultValue="learn" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800">
          <TabsTrigger value="learn">Learn</TabsTrigger>
          <TabsTrigger value="visualize">Visualize</TabsTrigger>
          <TabsTrigger value="code">Python Code</TabsTrigger>
        </TabsList>

        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Database className="w-6 h-6 text-emerald-400" />
                What is RAG?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-emerald-400">RAG (Retrieval Augmented Generation)</strong> is a technique 
                that gives LLMs access to external knowledge. Instead of relying only on what the model learned 
                during training, RAG retrieves relevant information from your documents to answer questions!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-emerald-500/30">
                <h4 className="text-emerald-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Imagine taking an open-book exam. Instead of memorizing everything, you can look up 
                information in your textbook. RAG lets AI do the same - it searches through documents 
                to find relevant information before answering!</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Why Use RAG?</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { problem: 'Knowledge Cutoff', solution: 'Access up-to-date information', icon: '📅' },
                  { problem: 'Hallucinations', solution: 'Ground answers in real documents', icon: '🎯' },
                  { problem: 'Generic Answers', solution: 'Use your specific data', icon: '📚' },
                  { problem: 'Privacy Concerns', solution: 'Keep data in your control', icon: '🔒' },
                ].map((item, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-2xl">{item.icon}</span>
                      <div>
                        <p className="text-red-400 text-sm line-through">{item.problem}</p>
                        <p className="text-green-400 font-medium">{item.solution}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">The RAG Pipeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                {[
                  { step: '1. Chunk', desc: 'Split docs into pieces', icon: '✂️', color: 'bg-blue-500' },
                  { step: '2. Embed', desc: 'Convert to vectors', icon: '🔢', color: 'bg-purple-500' },
                  { step: '3. Store', desc: 'Save in vector DB', icon: '💾', color: 'bg-green-500' },
                  { step: '4. Retrieve', desc: 'Find relevant docs', icon: '🔍', color: 'bg-orange-500' },
                  { step: '5. Generate', desc: 'LLM answers', icon: '✨', color: 'bg-pink-500' },
                ].map((item, i) => (
                  <div key={i} className="flex flex-col items-center">
                    <div className={`w-16 h-16 ${item.color} rounded-full flex items-center justify-center text-2xl mb-2`}>
                      {item.icon}
                    </div>
                    <p className="text-white font-semibold text-sm">{item.step}</p>
                    <p className="text-slate-400 text-xs text-center">{item.desc}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Search className="w-5 h-5 text-emerald-400" />
                Interactive RAG Demo
              </CardTitle>
              <CardDescription className="text-slate-400">
                See how RAG retrieves and generates answers!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask about Python, JavaScript, ML, React, or TensorFlow..."
                  className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                />
                <Button onClick={simulateRAG} className="bg-emerald-600 hover:bg-emerald-700">
                  Search
                </Button>
              </div>

              <div className="grid md:grid-cols-3 gap-4">
                <div className={`bg-slate-900 p-4 rounded-lg border-2 transition-all ${step >= 1 ? 'border-blue-500' : 'border-slate-700'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <Search className={`w-5 h-5 ${step >= 1 ? 'text-blue-400' : 'text-slate-500'}`} />
                    <h4 className="text-white font-semibold">1. Query</h4>
                  </div>
                  <p className="text-slate-400 text-sm">{query || 'Enter a question...'}</p>
                </div>

                <div className={`bg-slate-900 p-4 rounded-lg border-2 transition-all ${step >= 2 ? 'border-purple-500' : 'border-slate-700'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <FileText className={`w-5 h-5 ${step >= 2 ? 'text-purple-400' : 'text-slate-500'}`} />
                    <h4 className="text-white font-semibold">2. Retrieved</h4>
                  </div>
                  {retrievedDocs.length > 0 ? (
                    <div className="space-y-1">
                      {retrievedDocs.map((doc, i) => (
                        <p key={i} className="text-slate-400 text-xs bg-slate-800 p-2 rounded">{doc}</p>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm">Waiting for retrieval...</p>
                  )}
                </div>

                <div className={`bg-slate-900 p-4 rounded-lg border-2 transition-all ${step >= 3 ? 'border-green-500' : 'border-slate-700'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className={`w-5 h-5 ${step >= 3 ? 'text-green-400' : 'text-slate-500'}`} />
                    <h4 className="text-white font-semibold">3. Answer</h4>
                  </div>
                  <p className={`text-sm ${step >= 3 ? 'text-green-300' : 'text-slate-500'}`}>
                    {answer || 'Waiting for generation...'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Vector Similarity Visualization</CardTitle>
              <CardDescription className="text-slate-400">
                How embeddings find similar content
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative h-64 bg-slate-900 rounded-lg p-4">
                <svg className="w-full h-full" viewBox="0 0 400 200">
                  {/* Query point */}
                  <circle cx="200" cy="100" r="12" fill="#10b981" />
                  <text x="200" y="130" textAnchor="middle" fill="#10b981" fontSize="10">Query</text>
                  
                  {/* Document points */}
                  <circle cx="180" cy="80" r="8" fill="#3b82f6" />
                  <text x="180" y="65" textAnchor="middle" fill="#3b82f6" fontSize="8">Doc 1 (0.95)</text>
                  
                  <circle cx="220" cy="90" r="8" fill="#3b82f6" />
                  <text x="220" y="75" textAnchor="middle" fill="#3b82f6" fontSize="8">Doc 2 (0.89)</text>
                  
                  <circle cx="170" cy="120" r="8" fill="#3b82f6" />
                  <text x="170" y="140" textAnchor="middle" fill="#3b82f6" fontSize="8">Doc 3 (0.82)</text>
                  
                  {/* Distant documents */}
                  <circle cx="80" cy="50" r="6" fill="#6b7280" />
                  <circle cx="320" cy="150" r="6" fill="#6b7280" />
                  <circle cx="350" cy="60" r="6" fill="#6b7280" />
                  
                  {/* Connection lines */}
                  <line x1="200" y1="100" x2="180" y2="80" stroke="#10b981" strokeWidth="2" strokeDasharray="4" />
                  <line x1="200" y1="100" x2="220" y2="90" stroke="#10b981" strokeWidth="2" strokeDasharray="4" />
                  <line x1="200" y1="100" x2="170" y2="120" stroke="#10b981" strokeWidth="2" strokeDasharray="4" />
                </svg>
              </div>
              <p className="text-slate-400 text-sm mt-2 text-center">
                Documents closest to the query vector are retrieved. Numbers show similarity scores.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Popular Vector Databases</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  { name: 'Chroma', type: 'Local', desc: 'Simple, great for prototyping', color: 'bg-yellow-500' },
                  { name: 'Pinecone', type: 'Cloud', desc: 'Scalable, managed service', color: 'bg-blue-500' },
                  { name: 'FAISS', type: 'Local', desc: 'Fast, by Meta/Facebook', color: 'bg-purple-500' },
                  { name: 'Weaviate', type: 'Both', desc: 'Feature-rich, open-source', color: 'bg-green-500' },
                  { name: 'Milvus', type: 'Both', desc: 'Highly scalable', color: 'bg-orange-500' },
                  { name: 'Qdrant', type: 'Both', desc: 'Rust-based, fast', color: 'bg-red-500' },
                ].map((db, i) => (
                  <div key={i} className="bg-slate-900 p-3 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <div className={`w-6 h-6 ${db.color} rounded flex items-center justify-center`}>
                        <Database className="w-3 h-3 text-white" />
                      </div>
                      <span className="text-white font-medium">{db.name}</span>
                    </div>
                    <Badge variant="outline" className="text-slate-400 text-xs mb-1">{db.type}</Badge>
                    <p className="text-slate-500 text-xs">{db.desc}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-emerald-400" />
                Python Code: Building RAG Systems
              </CardTitle>
              <CardDescription className="text-slate-400">
                From simple to advanced RAG implementations!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
                <h4 className="text-emerald-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install langchain chromadb faiss-cpu openai llama-index
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module11RAG
