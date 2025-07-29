import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langsmith import Client
import os
from dotenv import load_dotenv


load_dotenv()

# Set the LangSmith (LangChain observability) API key and project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_v2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Project"

# Initialize the LangChain tracer linked to a specific project
client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))


# Initialize a wrapper utility around the Wikipedia API and a query tool that uses the defined Wikipedia API wrapper
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250) 
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Initialize a wrapper utility around the Arxiv API and a  query tool that uses the defined arxiv API wrapper
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name= 'Search')


# Title for the app
st.title("LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")


# Check if the "messages" variable is already present in the Streamlit session state
# Session state is used to persist information (like chat history) between reruns of the app
if "messages" not in st.session_state:
    
    # If not present, initialize "messages" as a list with a default assistant greeting
    # This ensures the chat always starts with an assistant message for new sessions
    st.session_state["messages"] = [
        {
            "role": "assistant",   # Who sent the message (assistant or user)
            "content": "Hi, I am a chatbot that can search the web. How can I help you?"  # Initial assistant message
        }
    ]

# Iterate through all messages stored in session state (the conversation history)
for message in st.session_state.messages:
    
    # Display each message in the chat window using the appropriate role (assistant or user)
    # .write() renders the content of the message
    st.chat_message(message['role']).write(message['content'])


# Prompt the user for input using Streamlit's chat input widget
# The walrus operator (:=) assigns the user's input to the variable 'prompt' and checks if it is not empty or None
if prompt := st.chat_input(placeholder="What is machine learning?"):
    
    # If the user submitted a message, append it to the chat history stored in session state
    # This keeps a record of the user's messages in the ongoing conversation
    st.session_state.messages.append(
        {
            "role": "user",   # Specify that this message is from the user
            "content": prompt # Store the user's input text
        }
    )

    # Display the user's message in the chat UI, styled as a user message
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key = api_key,
        model_name = "Llama-3.3-70b-Versatile",
        streaming=True   # model sends the response in small chunks as it's generated, for real-time output
    )

    
    tools = [wiki, arxiv, search]

    # Create a LangChain agent that can use tools and reason about which one to use, step by step
    # - tools: list of available tools (e.g., search, wiki, arxiv)
    # - llm: the language model the agent uses for reasoning and response generation
    # - agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION: use the zero-shot ReAct agent type, which means
    #   the agent will reason and decide on actions (tool use) without being shown examples (zero-shot)
    # - handling_parsing_errors=True: enables the agent to gracefully handle and recover from output parsing errors
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append(
            {
                "role" : "assistant",
                "content": response
            }
        )

        st.write(str(response))






