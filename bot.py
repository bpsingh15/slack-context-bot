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

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    verbose=False
)

@app.event("app_mention")
def handle_mention(event, say):
    """Handle when the bot is mentioned"""
    question = event['text']
    
    # Remove bot mention from question
    question = question.split('>', 1)[-1].strip()
    
    if not question:
        say("Hi! Ask me anything about our Slack history! 👋")
        return
    
    # Show typing indicator
    say("Searching through our conversation history... 🔍")
    
    try:
        # Get answer from chain
        result = qa_chain({"question": question})
        answer = result['answer']
        source_docs = result.get('source_documents', [])
        
        # Format response with sources
        response = f"*Answer:*\n{answer}\n\n"
        
        if source_docs:
            response += "*📎 Sources:*\n"
            seen_links = set()
            for i, doc in enumerate(source_docs[:3], 1):  # Top 3 sources
                metadata = doc.metadata
                link = metadata.get('link', '')
                channel = metadata.get('channel', 'unknown')
                
                if link and link not in seen_links:
                    response += f"{i}. <{link}|#{channel} message>\n"
                    seen_links.add(link)
        
        say(response)
        
    except Exception as e:
        say(f"Sorry, I encountered an error: {str(e)}")
        print(f"Error: {e}")

@app.message("hello")
def handle_hello(message, say):
    """Respond to hello messages"""
    say(f"Hey there <@{message['user']}>! 👋")

if __name__ == "__main__":
    print("⚡ Bot is running!")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()