import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool

# Load secrets from the secrets.toml file
load_dotenv()
tavily_api_key = st.secrets["TAVILY_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
llama3_base_url = st.secrets["llama3"]["base_url"]
llama3_api_key = st.secrets["llama3"]["api_key"]
llama3_model = st.secrets["llama3"]["model"]

# Set environment variables if needed
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Disable LangChain Tracing if not needed
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Initialize search tool
search = TavilySearchResults(k=5)

# Initialize Llama3 model
llama3 = ChatOpenAI(
    base_url=llama3_base_url,
    api_key=llama3_api_key,
    model=llama3_model,
    temperature=0.1,
)

# Load prompts
json_prompt = hub.pull("hwchase17/react-chat-json")
prompt_ko = hub.pull("teddynote/react-chat-json-korean")

# Load PDF from GitHub URL
pdf_url = "https://github.com/your-username/your-repo/raw/main/라이엇게임즈.pdf"
pdf_response = requests.get(pdf_url)

# Save PDF locally
with open("라이엇게임즈.pdf", "wb") as pdf_file:
    pdf_file.write(pdf_response.content)

# Load PDF and create retriever
loader = PyPDFLoader("라이엇게임즈.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = loader.load_and_split(text_splitter)
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
retriever = vector.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="[라이엇게임즈] 관련 정보를 PDF 문서에서 검색합니다. '[라이엇게임즈]' 과 관련된 질문은 이 도구를 사용해야 합니다!",
)

# Define tools
tools = [search, retriever_tool]

# Create agents
llama3_agent = create_json_chat_agent(llama3, tools, json_prompt)
llama3_agent_ko = create_json_chat_agent(llama3, tools, prompt_ko)

# Create agent executors
llama3_agent_executor = AgentExecutor(
    agent=llama3_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

llama3_agent_executor_ko = AgentExecutor(
    agent=llama3_agent_ko,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# Streamlit app
st.title("Streamlit Chatbot")

st.write("Ask me anything!")

user_input = st.text_input("Your question:")

if user_input:
    response = llama3_agent_executor_ko.invoke(
        {"input": user_input}
    )
    st.write(f'Answer: {response["output"]}')

    # Display URLs from the response if available
    if "sources" in response:
        st.write("References:")
        for source in response["sources"]:
            st.write(f"{source['title']}: {source['url']}")
