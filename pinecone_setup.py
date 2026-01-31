from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
import time

def setup_pinecone():
    """Initialize Pinecone and create index if it doesn't exist"""
    # Initialize Pinecone with new API
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get list of existing indexes
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    # Check if index exists
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(10)
        print("Index created successfully!")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists")
    
    # Return the index
    return pc.Index(PINECONE_INDEX_NAME)

if __name__ == "__main__":
    setup_pinecone()
    print("Pinecone setup complete!")