#!/usr/bin/env python3
"""
Complete RAG (Retrieval-Augmented Generation) Pipeline with vLLM on SageMaker

This example demonstrates:
1. Document ingestion and embedding
2. Vector database storage (Pinecone/FAISS)
3. Query processing and retrieval
4. LLM inference with vLLM on SageMaker
5. Response generation with citations

Architecture:
    Documents → Embedding Model → Vector DB (Pinecone)
                                      ↓
    User Query → Retrieval → Top-K Chunks
                                ↓
                         vLLM (SageMaker) → Response + Citations
"""

import boto3
import json
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: Document Ingestion and Embedding
# ============================================================================

class DocumentEmbedder:
    """Embed documents using SageMaker embedding endpoint"""

    def __init__(self, embedding_endpoint: str):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.endpoint_name = embedding_endpoint

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps({"inputs": text})
        )

        result = json.loads(response['Body'].read())
        return np.array(result['embeddings'][0])

    def embed_documents(self, documents: List[str], chunk_size: int = 512) -> List[Dict]:
        """
        Split documents into chunks and embed them

        Args:
            documents: List of document texts
            chunk_size: Max tokens per chunk

        Returns:
            List of dicts with: {text, embedding, metadata}
        """
        chunks = []

        for doc_id, document in enumerate(documents):
            # Simple chunking (can use LangChain's text splitters for advanced chunking)
            words = document.split()

            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i + chunk_size])

                if len(chunk_text.strip()) < 50:  # Skip very small chunks
                    continue

                # Generate embedding
                embedding = self.embed_text(chunk_text)

                chunks.append({
                    "id": f"doc_{doc_id}_chunk_{i//chunk_size}",
                    "text": chunk_text,
                    "embedding": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_id": i // chunk_size,
                        "source": f"document_{doc_id}",
                    }
                })

                logger.info(f"Embedded chunk {doc_id}:{i//chunk_size}")

        return chunks


# ============================================================================
# STEP 2: Vector Database Storage
# ============================================================================

class VectorStore:
    """Simple in-memory vector store (can replace with Pinecone/Weaviate/FAISS)"""

    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.texts = []

    def add_documents(self, chunks: List[Dict]):
        """Add embedded chunks to vector store"""
        for chunk in chunks:
            self.vectors.append(chunk["embedding"])
            self.texts.append(chunk["text"])
            self.metadata.append(chunk["metadata"])

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Find top-k most similar documents

        Returns:
            List of (text, similarity_score, metadata) tuples
        """
        if len(self.vectors) == 0:
            return []

        # Compute cosine similarity
        vectors_array = np.array(self.vectors)
        similarities = np.dot(vectors_array, query_embedding) / (
            np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.texts[i], float(similarities[i]), self.metadata[i])
            for i in top_indices
        ]

        return results


# Alternative: Pinecone Integration
class PineconeVectorStore:
    """Production-grade vector store using Pinecone"""

    def __init__(self, api_key: str, environment: str, index_name: str):
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. Run: pip install pinecone-client")

        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def add_documents(self, chunks: List[Dict]):
        """Add embedded chunks to Pinecone"""
        vectors_to_upsert = [
            (
                chunk["id"],
                chunk["embedding"].tolist(),
                {"text": chunk["text"], **chunk["metadata"]}
            )
            for chunk in chunks
        ]

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)

        logger.info(f"Upserted {len(chunks)} vectors to Pinecone")

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Query Pinecone for similar documents"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        return [
            (match['metadata']['text'], match['score'], match['metadata'])
            for match in results['matches']
        ]


# ============================================================================
# STEP 3: vLLM Integration on SageMaker
# ============================================================================

class vLLMGenerator:
    """Generate responses using vLLM on SageMaker"""

    def __init__(self, endpoint_name: str):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.endpoint_name = endpoint_name

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response from vLLM endpoint"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True,
            }
        }

        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read())
        return result[0]["generated_text"]


# ============================================================================
# STEP 4: RAG Pipeline Orchestration
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline orchestration"""

    def __init__(
        self,
        embedding_endpoint: str,
        llm_endpoint: str,
        vector_store: VectorStore,
    ):
        self.embedder = DocumentEmbedder(embedding_endpoint)
        self.llm = vLLMGenerator(llm_endpoint)
        self.vector_store = vector_store

    def ingest_documents(self, documents: List[str]):
        """Ingest and embed documents"""
        logger.info("Ingesting documents...")
        chunks = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(chunks)
        logger.info(f"✓ Ingested {len(documents)} documents ({len(chunks)} chunks)")

    def query(self, question: str, top_k: int = 5) -> Dict[str, any]:
        """
        Process a query through RAG pipeline

        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve

        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Processing query: {question}")

        # 1. Embed the question
        logger.info("Embedding question...")
        question_embedding = self.embedder.embed_text(question)

        # 2. Retrieve relevant chunks
        logger.info(f"Retrieving top {top_k} relevant chunks...")
        relevant_chunks = self.vector_store.similarity_search(question_embedding, top_k=top_k)

        # 3. Build context from retrieved chunks
        context_parts = []
        sources = []

        for i, (text, score, metadata) in enumerate(relevant_chunks):
            context_parts.append(f"[{i+1}] {text}")
            sources.append({
                "chunk_id": i+1,
                "source": metadata.get("source", "unknown"),
                "similarity_score": score,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
            })

        context = "\n\n".join(context_parts)

        # 4. Build prompt with retrieved context
        prompt = self._build_rag_prompt(question, context)

        logger.info("Generating response with vLLM...")

        # 5. Generate answer
        answer = self.llm.generate(prompt, max_tokens=512, temperature=0.7)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context": context,
        }

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build RAG prompt with retrieved context"""
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer: Based on the provided context, """

        return prompt


# ============================================================================
# STEP 5: Deployment Example
# ============================================================================

def deploy_embedding_endpoint():
    """Deploy embedding model (e.g., all-MiniLM-L6-v2)"""
    from sagemaker.huggingface import HuggingFaceModel

    embedding_model = HuggingFaceModel(
        model_data="s3://your-bucket/models/all-MiniLM-L6-v2.tar.gz",
        role="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.1-transformers4.37-cpu-py310",
        env={
            "HF_MODEL_ID": "sentence-transformers/all-MiniLM-L6-v2",
            "HF_TASK": "feature-extraction",
        }
    )

    predictor = embedding_model.deploy(
        instance_type="ml.m5.xlarge",  # CPU is fine for embeddings
        initial_instance_count=1,
        endpoint_name="embedding-model",
    )

    return predictor.endpoint_name


def deploy_vllm_endpoint():
    """Deploy vLLM endpoint for generation"""
    from sagemaker import Model

    vllm_model = Model(
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.11.2-gpu-sagemaker",
        role="arn:aws:iam::123456789012:role/SageMakerRole",
        env={
            "SM_VLLM_MODEL": "meta-llama/Llama-3-8b-Instruct",
            "SM_VLLM_TENSOR_PARALLEL_SIZE": "1",
            "SM_VLLM_MAX_MODEL_LEN": "8192",
            "HF_TOKEN": "hf_...",
        }
    )

    predictor = vllm_model.deploy(
        instance_type="ml.g5.2xlarge",
        initial_instance_count=1,
        endpoint_name="vllm-llama3-8b",
    )

    return predictor.endpoint_name


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Complete RAG pipeline example"""

    # Sample documents (replace with your actual documents)
    documents = [
        """Amazon Web Services (AWS) is a comprehensive cloud computing platform provided by Amazon.
        It offers over 200 services including compute, storage, databases, analytics, machine learning,
        and more. AWS was launched in 2006 and has become the world's most comprehensive and broadly adopted cloud platform.""",

        """AWS SageMaker is a fully managed machine learning service that enables developers and data scientists
        to quickly build, train, and deploy machine learning models at scale. It provides built-in algorithms,
        support for popular frameworks like PyTorch and TensorFlow, and managed infrastructure for training and hosting.""",

        """AWS Lambda is a serverless compute service that runs your code in response to events and automatically
        manages the underlying compute resources. You pay only for the compute time you consume. Lambda supports
        multiple programming languages including Python, Node.js, Java, and Go.""",
    ]

    # Initialize pipeline
    logger.info("Initializing RAG pipeline...")

    # Option 1: Use in-memory vector store (for testing)
    vector_store = VectorStore()

    # Option 2: Use Pinecone (for production)
    # vector_store = PineconeVectorStore(
    #     api_key="your-pinecone-api-key",
    #     environment="us-west1-gcp",
    #     index_name="rag-documents"
    # )

    pipeline = RAGPipeline(
        embedding_endpoint="embedding-model",  # Deploy with deploy_embedding_endpoint()
        llm_endpoint="vllm-llama3-8b",         # Deploy with deploy_vllm_endpoint()
        vector_store=vector_store,
    )

    # Ingest documents
    pipeline.ingest_documents(documents)

    # Query the RAG system
    questions = [
        "What is AWS SageMaker?",
        "How does AWS Lambda pricing work?",
        "When was AWS launched?",
    ]

    for question in questions:
        print("\n" + "="*80)
        result = pipeline.query(question, top_k=3)

        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  [{source['chunk_id']}] {source['source']} (similarity: {source['similarity_score']:.3f})")
            print(f"      {source['text_preview']}")
        print("="*80)


if __name__ == "__main__":
    # To deploy endpoints:
    # embedding_endpoint = deploy_embedding_endpoint()
    # vllm_endpoint = deploy_vllm_endpoint()

    # Then run the main RAG pipeline
    main()
