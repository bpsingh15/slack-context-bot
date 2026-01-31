import pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
import time

def setup_pinecone():
    """Initialize Pinecone and create index if it doesn't exist"""
    # Initialize Pinecone (old API style)
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-east-1-aws')
    
    # Get list of existing indexes
    existing_indexes = pinecone.list_indexes()
    
    # Check if index exists
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine'
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(10)
        print("Index created successfully!")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists")
    
    # Return the index
    return pinecone.Index(PINECONE_INDEX_NAME)

if __name__ == "__main__":
    setup_pinecone()
    print("Pinecone setup complete!")