# Vector Databases: Comprehensive Guide

Modern vector databases for similarity search, RAG systems, and AI applications. Complete guide covering leading platforms, implementation patterns, and best practices.

## Table of Contents

- [Overview](#overview)
- [Vector Database Platforms](#vector-database-platforms)
- [Embedding Strategies](#embedding-strategies)
- [Implementation Examples](#implementation-examples)
- [RAG Applications](#rag-applications)
- [Performance Optimization](#performance-optimization)

## Overview

Vector databases have become essential infrastructure for AI applications, enabling similarity search, recommendation systems, and retrieval-augmented generation (RAG).

### Key Capabilities

- **Similarity Search**: Find semantically similar content
- **Hybrid Search**: Combine vector and keyword search
- **Real-time Updates**: Dynamic index management
- **Scalability**: Handle billions of vectors
- **Metadata Filtering**: Combined vector/scalar queries

## Vector Database Platforms

### 1. Pinecone

```python
import pinecone
import openai
from sentence_transformers import SentenceTransformer
import numpy as np

class PineconeVectorDB:
    """Complete Pinecone integration with advanced features."""
    
    def __init__(self, api_key, environment, index_name):
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        self.index = None
        
    def create_index(self, dimension=1536, metric='cosine', pod_type='p1.x1'):
        """Create optimized Pinecone index."""
        
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                pod_type=pod_type,
                pods=1,
                replicas=1,
                metadata_config={
                    "indexed": ["category", "source", "timestamp"]
                }
            )
        
        self.index = pinecone.Index(self.index_name)
        return self.index
    
    def upsert_vectors(self, documents, embeddings, metadata_list=None, batch_size=100):
        """Batch upsert with metadata."""
        
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_data = {
                'id': f'doc_{i}',
                'values': embedding.tolist(),
                'metadata': {
                    'text': doc,
                    'length': len(doc),
                    **(metadata_list[i] if metadata_list else {})
                }
            }
            vectors.append(vector_data)
            
            # Batch upsert
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []
        
        # Upsert remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
    
    def hybrid_search(self, query_embedding, text_filter=None, 
                     metadata_filter=None, top_k=10):
        """Hybrid search with metadata filtering."""
        
        filter_dict = {}
        
        if text_filter:
            filter_dict['text'] = {'$regex': text_filter}
        
        if metadata_filter:
            filter_dict.update(metadata_filter)
        
        results = self.index.query(
            vector=query_embedding.tolist(),
            filter=filter_dict if filter_dict else None,
            top_k=top_k,
            include_metadata=True
        )
        
        return results
    
    def semantic_search_with_reranking(self, query, top_k=50, rerank_k=10):
        """Semantic search with reranking."""
        
        # Initial retrieval
        query_embedding = self.encode_text(query)
        initial_results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Rerank using cross-encoder
        reranked = self.rerank_results(query, initial_results.matches)
        
        return reranked[:rerank_k]
    
    def rerank_results(self, query, results):
        """Rerank results using cross-encoder."""
        
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            query_doc_pairs = []
            for result in results:
                doc_text = result.metadata.get('text', '')
                query_doc_pairs.append([query, doc_text])
            
            scores = reranker.predict(query_doc_pairs)
            
            # Combine with original scores
            for i, result in enumerate(results):
                result.rerank_score = scores[i]
            
            # Sort by rerank score
            return sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
        except ImportError:
            print("CrossEncoder not available, returning original results")
            return results

# Usage example
def setup_pinecone_system():
    """Complete Pinecone setup example."""
    
    # Initialize
    db = PineconeVectorDB(
        api_key="your-api-key",
        environment="your-env",
        index_name="semantic-search"
    )
    
    # Create index
    db.create_index(dimension=384)  # For sentence-transformers
    
    # Sample documents
    documents = [
        "Python is a programming language",
        "Machine learning models need training data",
        "Vector databases enable similarity search"
    ]
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    
    # Upsert with metadata
    metadata = [
        {"category": "programming", "difficulty": "beginner"},
        {"category": "ai", "difficulty": "intermediate"},
        {"category": "database", "difficulty": "advanced"}
    ]
    
    db.upsert_vectors(documents, embeddings, metadata)
    
    return db
```

### 2. Weaviate

```python
import weaviate
from weaviate.classes.init import Auth
import requests

class WeaviateVectorDB:
    """Comprehensive Weaviate implementation."""
    
    def __init__(self, url, api_key=None):
        auth_config = Auth.api_key(api_key) if api_key else None
        self.client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=auth_config
        )
    
    def create_schema(self, class_name="Document"):
        """Create optimized schema for documents."""
        
        schema = {
            "class": class_name,
            "description": "Document storage with vector search",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                },
                "generative-openai": {
                    "model": "gpt-3.5-turbo"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Document title"
                },
                {
                    "name": "category",
                    "dataType": ["string"],
                    "description": "Document category"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata"
                },
                {
                    "name": "embedding",
                    "dataType": ["number[]"],
                    "description": "Custom embedding vector"
                }
            ]
        }
        
        try:
            self.client.schema.create_class(schema)
            print(f"Schema created for class: {class_name}")
        except Exception as e:
            print(f"Schema creation failed: {e}")
    
    def batch_import(self, documents, class_name="Document", batch_size=100):
        """Efficient batch import with automatic vectorization."""
        
        with self.client.batch as batch:
            batch.batch_size = batch_size
            
            for i, doc in enumerate(documents):
                properties = {
                    "content": doc.get("content", ""),
                    "title": doc.get("title", f"Document {i}"),
                    "category": doc.get("category", "general"),
                    "metadata": doc.get("metadata", {})
                }
                
                # Add custom embedding if provided
                if "embedding" in doc:
                    properties["embedding"] = doc["embedding"]
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=class_name,
                    uuid=doc.get("id")
                )
    
    def semantic_search(self, query, class_name="Document", limit=10, 
                       where_filter=None, with_generate=False):
        """Advanced semantic search with optional generation."""
        
        search_query = (
            self.client.query
            .get(class_name, ["content", "title", "category", "metadata"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .with_additional(["certainty", "distance"])
        )
        
        # Add filters
        if where_filter:
            search_query = search_query.with_where(where_filter)
        
        # Add generation
        if with_generate:
            search_query = search_query.with_generate(
                single_prompt="Summarize this document: {content}"
            )
        
        results = search_query.do()
        return results
    
    def hybrid_search(self, query, class_name="Document", alpha=0.7, limit=10):
        """Hybrid search combining vector and keyword search."""
        
        results = (
            self.client.query
            .get(class_name, ["content", "title", "category"])
            .with_hybrid(
                query=query,
                alpha=alpha  # 0.0 = pure keyword, 1.0 = pure vector
            )
            .with_limit(limit)
            .with_additional(["score"])
            .do()
        )
        
        return results
    
    def rag_generation(self, query, class_name="Document", context_limit=5):
        """RAG-based text generation."""
        
        # Retrieve relevant context
        context_results = self.semantic_search(
            query, class_name, limit=context_limit
        )
        
        # Extract context
        contexts = []
        for result in context_results['data']['Get'][class_name]:
            contexts.append(result['content'])
        
        context_text = "\n".join(contexts)
        
        # Generate response
        generation_results = (
            self.client.query
            .get(class_name, ["content"])
            .with_near_text({"concepts": [query]})
            .with_generate(
                grouped_task=f"Based on the following context: {context_text}\n\nAnswer this question: {query}"
            )
            .with_limit(1)
            .do()
        )
        
        return generation_results
    
    def create_custom_vector_index(self, vectors, metadata_list, class_name="CustomVector"):
        """Create index with custom vectors."""
        
        # Create schema for custom vectors
        custom_schema = {
            "class": class_name,
            "description": "Custom vector storage",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"]
                },
                {
                    "name": "metadata",
                    "dataType": ["object"]
                }
            ]
        }
        
        try:
            self.client.schema.create_class(custom_schema)
        except:
            pass  # Class might already exist
        
        # Batch import with custom vectors
        with self.client.batch as batch:
            for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
                batch.add_data_object(
                    data_object={
                        "content": metadata.get("content", ""),
                        "metadata": metadata
                    },
                    class_name=class_name,
                    vector=vector.tolist()
                )
```

### 3. Chroma

```python
import chromadb
from chromadb.config import Settings
import numpy as np

class ChromaVectorDB:
    """Production-ready Chroma implementation."""
    
    def __init__(self, persist_directory="./chroma_db", host=None, port=None):
        if host and port:
            # Remote client
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local persistent client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        
        self.collections = {}
    
    def create_collection(self, name, embedding_function=None, metadata=None):
        """Create collection with custom embedding function."""
        
        try:
            collection = self.client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata=metadata or {"description": f"Collection {name}"}
            )
            self.collections[name] = collection
            return collection
        except Exception as e:
            if "already exists" in str(e):
                collection = self.client.get_collection(name)
                self.collections[name] = collection
                return collection
            raise e
    
    def add_documents(self, collection_name, documents, embeddings=None, 
                     metadata_list=None, ids=None):
        """Add documents with optional custom embeddings."""
        
        collection = self.collections.get(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Prepare metadata
        if metadata_list is None:
            metadata_list = [{"source": "unknown"} for _ in documents]
        
        # Add to collection
        if embeddings is not None:
            collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadata_list,
                ids=ids
            )
        else:
            collection.add(
                documents=documents,
                metadatas=metadata_list,
                ids=ids
            )
    
    def similarity_search(self, collection_name, query, n_results=10, 
                         where=None, where_document=None, include=None):
        """Advanced similarity search with filtering."""
        
        collection = self.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        results = collection.query(
            query_texts=[query] if isinstance(query, str) else query,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include or ["documents", "metadatas", "distances"]
        )
        
        return results
    
    def similarity_search_with_embeddings(self, collection_name, query_embedding, 
                                        n_results=10, where=None):
        """Search using pre-computed embeddings."""
        
        collection = self.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def update_documents(self, collection_name, ids, documents=None, 
                        embeddings=None, metadata_list=None):
        """Update existing documents."""
        
        collection = self.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        update_data = {"ids": ids}
        
        if documents:
            update_data["documents"] = documents
        if embeddings is not None:
            update_data["embeddings"] = embeddings.tolist()
        if metadata_list:
            update_data["metadatas"] = metadata_list
        
        collection.update(**update_data)
    
    def delete_documents(self, collection_name, ids=None, where=None):
        """Delete documents by IDs or filter."""
        
        collection = self.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        if ids:
            collection.delete(ids=ids)
        elif where:
            collection.delete(where=where)
        else:
            raise ValueError("Must provide either ids or where filter")
    
    def get_collection_stats(self, collection_name):
        """Get collection statistics."""
        
        collection = self.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        count = collection.count()
        
        # Sample to get embedding dimension
        sample = collection.peek(limit=1)
        dimension = len(sample['embeddings'][0]) if sample['embeddings'] else 0
        
        return {
            "name": collection_name,
            "count": count,
            "dimension": dimension,
            "metadata": collection.metadata
        }

# Advanced usage example
def setup_chroma_rag_system():
    """Setup RAG system with Chroma."""
    
    from sentence_transformers import SentenceTransformer
    
    # Initialize
    db = ChromaVectorDB(persist_directory="./rag_db")
    
    # Create embedding function
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embedding_function(texts):
        return model.encode(texts).tolist()
    
    # Create collection
    collection = db.create_collection(
        name="documents",
        embedding_function=embedding_function,
        metadata={"description": "RAG document collection"}
    )
    
    # Sample documents
    documents = [
        "Artificial intelligence is transforming industries",
        "Machine learning requires large datasets",
        "Vector databases enable semantic search"
    ]
    
    metadata = [
        {"category": "AI", "importance": "high"},
        {"category": "ML", "importance": "medium"},
        {"category": "DB", "importance": "high"}
    ]
    
    # Add documents
    db.add_documents("documents", documents, metadata_list=metadata)
    
    return db
```

## Embedding Strategies

### 1. Text Embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import openai

class EmbeddingManager:
    """Comprehensive embedding generation and management."""
    
    def __init__(self):
        self.models = {}
        self.cached_embeddings = {}
    
    def load_sentence_transformer(self, model_name='all-MiniLM-L6-v2'):
        """Load sentence transformer model."""
        if model_name not in self.models:
            self.models[model_name] = SentenceTransformer(model_name)
        return self.models[model_name]
    
    def load_custom_transformer(self, model_name='bert-base-uncased'):
        """Load custom transformer for embeddings."""
        if model_name not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            self.models[model_name] = (tokenizer, model)
        return self.models[model_name]
    
    def generate_text_embeddings(self, texts, model_name='all-MiniLM-L6-v2', 
                                normalize=True, batch_size=32):
        """Generate embeddings for text documents."""
        
        model = self.load_sentence_transformer(model_name)
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def generate_openai_embeddings(self, texts, model="text-embedding-ada-002"):
        """Generate embeddings using OpenAI API."""
        
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(
                model=model,
                input=text
            )
            embeddings.append(response['data'][0]['embedding'])
        
        return np.array(embeddings)
    
    def generate_custom_embeddings(self, texts, model_name='bert-base-uncased'):
        """Generate embeddings using custom transformer."""
        
        tokenizer, model = self.load_custom_transformer(model_name)
        
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(embedding.numpy())
        
        return np.array(embeddings)
    
    def chunk_text_for_embeddings(self, text, chunk_size=500, overlap=50):
        """Chunk large text for better embeddings."""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def create_document_embeddings(self, document, strategy='chunked'):
        """Create embeddings for documents using different strategies."""
        
        if strategy == 'chunked':
            # Chunk document and average embeddings
            chunks = self.chunk_text_for_embeddings(document)
            chunk_embeddings = self.generate_text_embeddings(chunks)
            return np.mean(chunk_embeddings, axis=0)
        
        elif strategy == 'hierarchical':
            # Create multiple levels of embeddings
            sentences = document.split('.')
            paragraphs = document.split('\n\n')
            
            sentence_embeddings = self.generate_text_embeddings(sentences)
            paragraph_embeddings = self.generate_text_embeddings(paragraphs)
            document_embedding = self.generate_text_embeddings([document])
            
            return {
                'document': document_embedding[0],
                'paragraphs': paragraph_embeddings,
                'sentences': sentence_embeddings
            }
        
        else:  # 'simple'
            return self.generate_text_embeddings([document])[0]
    
    def compute_similarity_matrix(self, embeddings1, embeddings2=None):
        """Compute similarity matrix between embeddings."""
        
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        # Normalize embeddings
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        return similarity_matrix
    
    def find_similar_documents(self, query_embedding, document_embeddings, 
                              top_k=5, threshold=0.7):
        """Find most similar documents to query."""
        
        similarities = np.dot(query_embedding, document_embeddings.T)
        
        # Get top-k similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Filter by threshold
        valid_results = []
        for idx, sim in zip(top_indices, top_similarities):
            if sim >= threshold:
                valid_results.append((idx, sim))
        
        return valid_results

# Usage example
def create_embedding_pipeline():
    """Create comprehensive embedding pipeline."""
    
    manager = EmbeddingManager()
    
    # Sample documents
    documents = [
        "Python is a versatile programming language used for web development, data science, and automation.",
        "Machine learning algorithms learn patterns from data to make predictions on new, unseen data.",
        "Vector databases store high-dimensional vectors and enable fast similarity search operations."
    ]
    
    # Generate embeddings using different strategies
    simple_embeddings = manager.generate_text_embeddings(documents)
    
    # Generate hierarchical embeddings for first document
    hierarchical = manager.create_document_embeddings(documents[0], strategy='hierarchical')
    
    # Find similar documents
    query = "What is machine learning?"
    query_embedding = manager.generate_text_embeddings([query])[0]
    
    similar_docs = manager.find_similar_documents(
        query_embedding, simple_embeddings, top_k=3
    )
    
    return manager, similar_docs
```

## RAG Applications

### Complete RAG System

```python
class RAGSystem:
    """Production-ready RAG system with advanced features."""
    
    def __init__(self, vector_db, llm_client, embedding_model=None):
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.conversation_memory = []
    
    def ingest_documents(self, documents, collection_name="knowledge_base", 
                        chunk_size=500, overlap=50):
        """Ingest documents into the vector database."""
        
        all_chunks = []
        all_metadata = []
        
        for doc_id, document in enumerate(documents):
            if isinstance(document, dict):
                content = document.get('content', '')
                metadata = document.get('metadata', {})
            else:
                content = document
                metadata = {}
            
            # Chunk document
            chunks = self._chunk_document(content, chunk_size, overlap)
            
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata = {
                    **metadata,
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'total_chunks': len(chunks)
                }
                all_metadata.append(chunk_metadata)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(all_chunks)
        
        # Store in vector database
        self.vector_db.add_documents(
            collection_name,
            all_chunks,
            embeddings=embeddings,
            metadata_list=all_metadata
        )
        
        return len(all_chunks)
    
    def retrieve_context(self, query, collection_name="knowledge_base", 
                        top_k=5, rerank=True):
        """Retrieve relevant context for query."""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search vector database
        results = self.vector_db.similarity_search_with_embeddings(
            collection_name, query_embedding, n_results=top_k * 2
        )
        
        # Extract context
        contexts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            contexts.append({
                'content': doc,
                'metadata': metadata,
                'score': 1 - distance,  # Convert distance to similarity
                'rank': i
            })
        
        # Rerank if requested
        if rerank:
            contexts = self._rerank_contexts(query, contexts)
        
        return contexts[:top_k]
    
    def generate_response(self, query, collection_name="knowledge_base", 
                         system_prompt=None, include_sources=True, 
                         conversation_aware=True):
        """Generate response using RAG."""
        
        # Retrieve context
        contexts = self.retrieve_context(query, collection_name)
        
        # Prepare context text
        context_text = "\n".join([
            f"[Source {i+1}]: {ctx['content']}" 
            for i, ctx in enumerate(contexts)
        ])
        
        # Prepare conversation history
        conversation_context = ""
        if conversation_aware and self.conversation_memory:
            recent_conversation = self.conversation_memory[-3:]  # Last 3 exchanges
            conversation_context = "\n".join([
                f"Human: {exchange['query']}\nAssistant: {exchange['response']}"
                for exchange in recent_conversation
            ])
        
        # Create prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use the context to provide accurate, helpful responses. If the context doesn't contain enough 
            information to answer the question, say so clearly."""
        
        prompt = f"""System: {system_prompt}

Conversation History:
{conversation_context}

Context:
{context_text} 