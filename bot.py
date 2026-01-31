from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize Pinecone and embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Create vectorstore using langchain-pinecone
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)

# ... rest of the file stays the same ...