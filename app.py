import os
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Streamlit Page Config
st.set_page_config(page_title="LawAID")

# Logo Display
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00")

# Custom CSS
st.markdown(
    """
    <style>
        div.stButton > button:first-child {
            background-color: #ffd0d0;
        }
        div.stButton > button:active {
            background-color: #ff6262;
        }
        div[data-testid="stStatusWidget"] div button {
            display: none;
        }
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        button[title="View fullscreen"] {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Reset Chat Memory
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Load HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
)

# Load or Create FAISS Vector Store
faiss_path = "faiss_index"
if os.path.exists(faiss_path):
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_texts(["sample legal document"], embeddings)
    db.save_local(faiss_path)

db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define Prompt Template
prompt_template = """<s>[INST] This is a chat template. As a legal chatbot specializing in Indian Penal Code queries, 
your objective is to provide accurate and concise information based on the user's questions. Do not generate 
your own questions and answers. You will strictly follow instructions, using only relevant legal context. 
Avoid unnecessary details. If a question falls outside the given context, you will use your own knowledge base. 
Do not ask additional questions. Your responses should be professional, precise, and compliant with the Indian Penal Code.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Get API Key Safely
TOGETHER_AI_API = os.environ.get("TOGETHER_AI", "941d4ab8ae26aaca46f5831501d9357f390a2432ccda9f359505fad0769b594d")

# Load LLM Model
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_AI_API
)

# Create Conversational Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# User Input
input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()
            full_response = "⚠ *Note: Information provided may be inaccurate.* \n\n\n"

        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02)
            message_placeholder.markdown(full_response + " ▌")
        
        st.button('Reset All Chat 🗑', on_click=reset_conversation)

# Sidebar for Contractual Complaint Form & Case Winning Probability
with st.sidebar:
    st.header("Contractual Complaint Form")

    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone Number")
    contract_type = st.selectbox("Type of Contract", ["Employment", "Lease", "NDA", "Service Agreement", "Other"])
    jurisdiction = st.text_input("Jurisdiction (Applicable Law)")
    breach_description = st.text_area("Describe the Issue/Breach of Contract")
    legal_assistance = st.radio("Do you need legal assistance?", ["Yes", "No"]) 
    supporting_docs = st.file_uploader("Upload Supporting Documents (PDF, DOCX, etc.)", type=["pdf", "docx"])

    submit = st.button("Submit Complaint")

    if submit:
        if not name or not email or not contract_type or not jurisdiction or not breach_description:
            st.warning("Please fill in all required fields.")
        else:
            st.success("Complaint registered successfully! A legal expert will review your case.")

    st.header("Case Winning Probability")
    probability_score = "Processing..."
    if breach_description:
        probability_score = "High" if "strong evidence" in breach_description else "Moderate"
    st.write(f"*Estimated Probability:* {probability_score}")
