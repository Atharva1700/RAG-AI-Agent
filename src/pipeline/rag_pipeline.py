"""
rag_pipeline.py
Phase 2: Core RAG chain - retrieval + generation.

Resume connection (Uber):
'Improving retrieval precision by 35% through groundedness
and context relevance evaluation'
This chain is what generates those answers.
"""

import os
import logging
import time
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

CHROMA_DIR  = "./chroma_db"
COLLECTION  = "uber_support_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "google/flan-t5-base"
TOP_K       = 4

# ── Prompt template ──────────────────────────────────────────────────
# This is the groundedness constraint.
# The LLM is told to use ONLY the context and say
# "I don't know" if the answer is not there.
PROMPT_TEMPLATE = """You are a helpful Uber support assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: I don't have enough information to answer that.
Keep your answer concise and accurate.

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB collection from disk."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            f"Run ingest.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )
    count = vectorstore._collection.count()
    logger.info(f"Loaded ChromaDB: {count} vectors in collection '{COLLECTION}'")
    return vectorstore


def build_llm() -> HuggingFacePipeline:
    """
    Build a local LLM using HuggingFace transformers.
    Uses flan-t5-base: free, runs on CPU, no API key needed.
    In production: swap for OpenAI, Anthropic, or any other LLM.
    """
    logger.info(f"Loading LLM: {LLM_MODEL} (first run downloads ~900MB)")
    pipe = hf_pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("LLM loaded successfully")
    return llm


def build_rag_chain(vectorstore: Chroma, llm: HuggingFacePipeline) -> RetrievalQA:
    """
    Assemble the complete RAG chain:
    retriever -> prompt -> LLM -> answer + sources
    """
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # stuff = concatenate all chunks into one prompt
        retriever=retriever,
        return_source_documents=True, # needed for evaluation + citations
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info(f"RAG chain built: top-{TOP_K} retrieval, flan-t5-base generation")
    return chain


def query_rag(
    chain: RetrievalQA,
    question: str,
    verbose: bool = True
) -> dict:
    """
    Run a single query through the RAG chain.
    Returns: answer, sources, latency_ms
    """
    start = time.time()
    result = chain.invoke({"query": question})
    latency_ms = round((time.time() - start) * 1000, 1)

    answer  = result["result"].strip()
    sources = [
        {
            "source":   doc.metadata.get("source", "unknown"),
            "category": doc.metadata.get("category", "unknown"),
            "snippet":  doc.page_content[:120] + "..."
        }
        for doc in result["source_documents"]
    ]

    if verbose:
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Latency: {latency_ms}ms")
        print(f"Sources used:")
        for s in sources:
            print(f"  [{s['source']}] {s['snippet']}")

    return {
        "question":   question,
        "answer":     answer,
        "sources":    sources,
        "latency_ms": latency_ms
    }


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TESTING RAG PIPELINE")
    print("="*50)

    vs    = load_vectorstore()
    llm   = build_llm()
    chain = build_rag_chain(vs, llm)

    test_questions = [
        "What is the cancellation fee for Uber rides?",
        "How do drivers get paid and how often?",
        "What safety features does Uber provide?",
        "What data does Uber collect and for how long?"
    ]

    for q in test_questions:
        print("\n" + "-"*50)
        query_rag(chain, q)

    print("\n" + "="*50)
    print("  RAG PIPELINE TEST COMPLETE")
    print("="*50)