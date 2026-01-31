from slack_sdk import WebClient
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from config import SLACK_BOT_TOKEN, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
import time
from datetime import datetime

class SlackIndexer:
    def __init__(self):
        self.slack_client = WebClient(token=SLACK_BOT_TOKEN)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)
    
    # ... rest of the class stays the same ...
        
    def get_all_channels(self):
        """Fetch all public channels"""
        try:
            response = self.slack_client.conversations_list(types="public_channel")
            return response['channels']
        except Exception as e:
            print(f"Error fetching channels: {e}")
            return []
    
    def get_channel_history(self, channel_id, limit=1000):
        """Fetch message history from a channel"""
        messages = []
        try:
            result = self.slack_client.conversations_history(
                channel=channel_id,
                limit=limit
            )
            messages = result['messages']
            
            # Handle pagination if there are more messages
            while result.get('has_more'):
                result = self.slack_client.conversations_history(
                    channel=channel_id,
                    cursor=result['response_metadata']['next_cursor'],
                    limit=limit
                )
                messages.extend(result['messages'])
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"Error fetching history for channel {channel_id}: {e}")
        
        return messages
    
    def format_message(self, message, channel_name, channel_id):
        """Format message for embedding"""
        text = message.get('text', '')
        user = message.get('user', 'Unknown')
        timestamp = message.get('ts', '')
        
        # Create readable timestamp
        if timestamp:
            dt = datetime.fromtimestamp(float(timestamp))
            readable_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            readable_time = 'Unknown time'
        
        # Create message link
        ts_clean = timestamp.replace('.', '')
        message_link = f"https://slack.com/app_redirect?channel={channel_id}&message_ts={ts_clean}"
        
        formatted = f"Channel: #{channel_name}\nUser: {user}\nTime: {readable_time}\nMessage: {text}"
        
        metadata = {
            'channel': channel_name,
            'channel_id': channel_id,
            'user': user,
            'timestamp': timestamp,
            'link': message_link,
            'text': text
        }
        
        return formatted, metadata
    
    def index_channel(self, channel_id, channel_name):
        """Index all messages from a channel"""
        print(f"\nIndexing channel: #{channel_name}")
        
        # Try to join the channel first
        try:
            self.slack_client.conversations_join(channel=channel_id)
            print(f"Joined #{channel_name}")
        except Exception as e:
            print(f"Could not join #{channel_name}: {e}")
        
        messages = self.get_channel_history(channel_id)
        
        print(f"Found {len(messages)} messages")
        
        # ... rest of the method stays the same ...
        
        batch_size = 100
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            vectors = []
            
            for msg in batch:
                # Skip bot messages and messages without text
                if msg.get('subtype') == 'bot_message' or not msg.get('text'):
                    continue
                
                formatted_msg, metadata = self.format_message(msg, channel_name, channel_id)
                
                # Create embedding
                try:
                    embedding = self.embeddings.embed_query(formatted_msg)
                    
                    # Create unique ID
                    vector_id = f"{channel_id}_{msg['ts']}"
                    
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': metadata
                    })
                except Exception as e:
                    print(f"Error embedding message: {e}")
                    continue
            
            # Upload batch to Pinecone
            if vectors:
                try:
                    self.index.upsert(vectors=vectors)
                    print(f"Indexed {len(vectors)} messages")
                except Exception as e:
                    print(f"Error upserting to Pinecone: {e}")
            
            time.sleep(1)  # Rate limiting
    
    def index_all_channels(self):
        """Index all public channels"""
        channels = self.get_all_channels()
        print(f"Found {len(channels)} channels")
        
        for channel in channels:
            self.index_channel(channel['id'], channel['name'])
        
        print("\n✅ Indexing complete!")

if __name__ == "__main__":
    print("Starting Slack indexing...")
    indexer = SlackIndexer()
    indexer.index_all_channels()