"""
ingest.py
Phase 1: Document ingestion, chunking, embedding, ChromaDB storage.

Resume connection (Uber):
'Built and validated RAG pipelines using vector libraries applying
ML retrieval model validation and QA protocols'
This file IS the ingestion half of that bullet.
"""

import os
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

CHROMA_DIR   = "./chroma_db"
COLLECTION   = "uber_support_docs"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50


def create_sample_documents() -> list[Document]:
    """
    Create Uber-domain sample documents.
    In production: replace with PyPDFLoader or DirectoryLoader
    loading real support tickets, policy docs, or product manuals.
    """
    raw_docs = [
        {
            "content": """Uber Driver Earnings and Payment Policy

Drivers on the Uber platform earn money by completing trips.
The base fare is calculated using a formula that includes:
- Base rate: a fixed amount per trip
- Time rate: earnings per minute while driving
- Distance rate: earnings per mile traveled
- Surge multiplier: applied during high-demand periods

Payments are processed weekly every Tuesday via direct deposit
to the driver's bank account. Instant Pay allows drivers to
cash out up to 5 times per day for a small fee of $0.50 per
cashout. Drivers must maintain a minimum rating of 4.6 stars
to remain active on the platform. Ratings below 4.6 trigger
an automatic review process where drivers receive coaching
resources and performance improvement plans.

Uber takes a service fee (commission) from each fare, typically
between 20-25% depending on the market and driver tier.
UberPro drivers in the Diamond tier receive reduced service fees
and access to additional benefits including vehicle discounts.""",
            "source": "driver_earnings_policy.txt",
            "category": "driver_policy"
        },
        {
            "content": """Uber Customer Refund and Billing Policy

Customers who experience issues with their Uber trips can request
refunds through the app or help center. Common refund scenarios:

1. Incorrect charges: If a customer is charged more than the
estimated fare shown at booking, they may request a fare review.
Uber's billing system automatically flags trips where the actual
fare exceeds the estimate by more than 20%.

2. Trip issues: Customers can report issues such as driver taking
a longer route, cleanliness problems, or safety concerns. Uber
reviews these reports within 24 hours and may issue partial or
full refunds based on the severity of the issue.

3. Cancellation fees: A cancellation fee of $5 is charged if the
customer cancels more than 2 minutes after the driver accepts
the trip. This fee is paid directly to the driver.

4. Lost items: Uber charges a $15 lost item fee if a driver
returns a lost item to the customer. This fee goes to the driver
as compensation for their time.

Refunds are typically processed within 3-5 business days.""",
            "source": "customer_billing_policy.txt",
            "category": "billing"
        },
        {
            "content": """Uber Safety Features and Emergency Protocols

Uber has several safety features built into the app:

RideCheck: Uber's technology detects unusual stops during a trip
and automatically checks in with both driver and rider. If there
is no response, Uber may contact emergency services.

Emergency Button: Both riders and drivers have access to a red
emergency button in the app that directly connects to local
emergency services and shares the trip's GPS location.

Share My Trip: Riders can share their real-time trip details
including the driver's name, photo, license plate, and GPS
location with trusted contacts.

Two-Factor Authentication: All driver accounts require phone
number verification. New drivers undergo a background check
that includes criminal history and driving record review.

Insurance Coverage: Uber provides liability insurance coverage
for all trips. During a trip, this includes up to $1 million
in third-party liability coverage. Between trips, drivers are
covered by their personal auto insurance.

Anonymous Communication: Phone numbers are masked during
communication between drivers and riders to protect privacy.""",
            "source": "safety_protocols.txt",
            "category": "safety"
        },
        {
            "content": """Uber Eats Delivery Partner Guidelines

Delivery partners on Uber Eats must follow specific guidelines
to maintain their active status on the platform:

Delivery Standards:
- Accept at least 50% of delivery requests during active hours
- Maintain a completion rate above 90%
- Keep customer rating above 4.5 stars
- Deliver orders within the estimated delivery window

Equipment Requirements:
- Insulated delivery bag required for all food orders
- Valid driver's license and vehicle registration
- Smartphone with Uber Driver app version 4.0 or higher

Earnings Structure:
Delivery partners earn a base delivery fee plus mileage
compensation for each completed order. Tips from customers
are paid 100% to delivery partners. During peak hours
(lunch 11am-2pm, dinner 5pm-9pm), a boost multiplier
increases earnings by 1.2x to 1.8x depending on demand.

Account Deactivation: Accounts may be deactivated for
maintaining completion rates below 80% for 4 consecutive
weeks, receiving multiple safety violations, or engaging
in fraudulent activity such as GPS spoofing.""",
            "source": "eats_delivery_guidelines.txt",
            "category": "delivery"
        },
        {
            "content": """Uber Data Privacy and GDPR Compliance

Uber collects and processes personal data in accordance with
global privacy regulations including GDPR in Europe and CCPA
in California. Key data practices:

Data Collected:
- Location data: GPS coordinates during trips
- Payment information: Encrypted card details via PCI-DSS
- Communications: Trip-related messages between drivers and riders
- Usage patterns: App interaction data for service improvement

Data Retention:
- Trip data is retained for 7 years for legal and tax purposes
- Deleted accounts: Personal data purged within 30 days
- Location data: Retained for 90 days after trip completion

User Rights (GDPR Article 15-20):
- Right to access: Request a copy of all personal data
- Right to erasure: Request deletion of personal data
- Right to portability: Export data in machine-readable format
- Right to rectification: Correct inaccurate personal data

Data Security: Uber uses AES-256 encryption for data at rest
and TLS 1.3 for data in transit. Annual third-party security
audits verify compliance with ISO 27001 standards.""",
            "source": "data_privacy_gdpr.txt",
            "category": "privacy"
        }
    ]

    documents = []
    for raw in raw_docs:
        doc = Document(
            page_content=raw["content"],
            metadata={
                "source":   raw["source"],
                "category": raw["category"]
            }
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} sample documents")
    return documents


def ingest_documents(documents: list[Document]) -> Chroma:
    """
    Chunk, embed, and store documents in ChromaDB.
    Returns the vectorstore object for immediate use.
    """
    # Step 1: Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    # Step 2: Load embedding model (downloads ~80MB first time)
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 3: Create ChromaDB collection and embed all chunks
    logger.info("Embedding chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION
    )

    logger.info(f"ChromaDB collection '{COLLECTION}' created")
    logger.info(f"Stored {len(chunks)} vectors in {CHROMA_DIR}/")
    return vectorstore


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RUNNING DOCUMENT INGESTION")
    print("="*50 + "\n")

    docs = create_sample_documents()
    vs   = ingest_documents(docs)

    # Verify: quick test query
    test_results = vs.similarity_search("driver payment cancellation fee", k=2)
    print("\n--- Verification Query ---")
    print("Query: 'driver payment cancellation fee'")
    for i, r in enumerate(test_results):
        print(f"Result {i+1}: [{r.metadata['source']}]")
        print(f"  {r.page_content[:100]}...")

    print("\n" + "="*50)
    print("  INGESTION COMPLETE")
    print(f"  Vectors stored: {CHROMA_DIR}/")
    print("="*50 + "\n")