# Import libraries

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Setup embeddings
os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HUGGINGFACE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# For Mistral AI
# mistral_key = os.getenv('MISTRALAI_API_KEY')
# os.environ['MISTRALAI_API_KEY'] = os.getenv('MISTRALAI_API_KEY')
# embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_key)

# Setup Streamlit page
st.set_page_config(page_title='DocuGenie', layout='wide')
st.title("📄 DocuGenie: Conversational Document Assistant")
st.caption("**Upload your PDFs and have interactive conversations with them!**")

# message = st.chat_message("assistant")
# message.write("**Hey, What can I help you with?**")

# Initialize session states
if 'sessions' not in st.session_state:
    st.session_state['sessions'] = [] 

if 'active_session' not in st.session_state:
    st.session_state['active_session'] = None

if 'groq_api_key' not in st.session_state:
    st.session_state['groq_api_key'] = ""

if 'session_files' not in st.session_state:
    st.session_state['session_files'] = {}

if 'session_histories' not in st.session_state:
    st.session_state['session_histories'] = {}

if 'session_responses' not in st.session_state:
    st.session_state['session_responses'] = {}

if 'session_user_inputs' not in st.session_state:
    st.session_state['session_user_inputs'] = {}

def reset_session_states(session_id):
    """Reset all states related to a session."""
    st.session_state['session_files'][session_id] = None
    st.session_state['session_histories'][session_id] = ChatMessageHistory()
    st.session_state['session_responses'][session_id] = []
    st.session_state['session_user_inputs'][session_id] = []

# Sidebar for settings
st.sidebar.title('⚙️ Settings')

# API Key Input
api_key = st.sidebar.text_input(
    "🔑 **Enter your Groq API Key:**",
    value=st.session_state['groq_api_key'],
    key="groq_api_key_input",
    type="password",
    help="Your API key will be stored securely for this session."
)

if api_key:
    st.session_state['groq_api_key'] = api_key

# Session Management
st.sidebar.subheader("🗂️ Session Management")

session_option = st.sidebar.selectbox(
    "**Select or start a session:**",
    ["➕ New Session"] + st.session_state['sessions'],
    index=0
)

if session_option == "➕ New Session":
    new_session_id = st.sidebar.text_input("**Enter a new Session ID:**", key="new_session_input")
    if st.sidebar.button("Start Session"):
        if new_session_id and new_session_id not in st.session_state['sessions']:
            st.session_state['sessions'].append(new_session_id)
            st.session_state['active_session'] = new_session_id
            reset_session_states(new_session_id)
            st.success(f"**Session '{new_session_id}' created and activated!**")
        else:
            st.sidebar.error("Session ID must be unique and non-empty.")
else:
    if session_option != st.session_state['active_session']:
        st.session_state['active_session'] = session_option
        if session_option not in st.session_state['session_files']:
            reset_session_states(session_option)
        st.success(f"Switched to session '{session_option}'.")

# CSS for styling question and answer
# css1 = """
#     <style>
#     .chat-message {
#         padding: 10px;
#         margin: 10px 0;
#         border-radius: 5px;
#         border: 1px solid #ddd;
#         background-color: #f9f9f9;
#     }
#     .user-message {
#         background-color: #d1e7dd;
#         text-align: left;
#     }
#     .ai-message {
#         background-color: #f8d7da;
#         text-align: left;
#     }
#     </style>
#     """
# st.markdown(css1, unsafe_allow_html=True)



# Main Content
if st.session_state['active_session']:
    session_id = st.session_state['active_session']
    st.markdown(f"**Active Session:** `{session_id}`")

    if not st.session_state['groq_api_key']:
        st.warning("Please enter your Groq API Key in the sidebar to proceed.")
    else:
        # Initialize the language model
        llm = ChatGroq(
            groq_api_key=st.session_state['groq_api_key'],
            model_name="Gemma2-9b-It"
        )

        # For Mistral AI

        # llm = ChatMistralAI(
        # #model = 'open-mixtral-8x7b',
        # #model = 'mistral-medium',
        # model="mistral-large-latest",
        # temperature=0.3,
        # api_key=mistral_key
        # )

        st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)

        # Upload PDFs
        st.subheader("📤 **Upload PDF Document**")
        uploaded_files = st.file_uploader(f"**Upload PDF files for session `{session_id}`:**",
            type="pdf",
            accept_multiple_files=True,
            key=f"file_uploader_{session_id}"
        )
        css = '''
            <style>
            [data-testid='stFileUploader'] {
                display: flex;
                align-items: center;
            }
            [data-testid='stFileUploader'] section {
                padding: 0;
            }
            [data-testid='stFileUploader'] section > input + div {
                display: none;
            }
            [data-testid='stFileUploader'] section + div {
                margin-left: 1cm; /* Adjust spacing for browse button */
                margin-right: auto; /* Push uploaded file name to the right */
            }
            </style>
            '''
        
        st.markdown(css, unsafe_allow_html=True)

        if uploaded_files:
            st.session_state['session_files'][session_id] = uploaded_files
            st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

            # Process Documents
            with st.spinner("Processing documents..."):
                documents = []
                for i in uploaded_files:
                    temppdf = f"./temp.pdf"
                    with open(temppdf, "wb") as file:
                        file.write(i.getvalue())
                        file_name=i.name
                        loader = PyPDFLoader(temppdf)
                        docs = loader.load()
                        documents.extend(docs)

                # Split and embed documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=300,
                    
                )
                docs_splits = text_splitter.split_documents(documents)
                vectorstore = Chroma.from_documents(
                    documents=docs_splits,
                    embedding=embeddings,
                    collection_name=f"{session_id}_collection"
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                st.success("Documents processed and ready for querying!")

            # Setup Prompts and Chains
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            system_prompt = (
                # "You are an assistant for question-answering tasks. "
                # "Use the following pieces of retrieved context to answer the user's question to teh best of your ability. "
                # "For the question, if you don't know the answer, say Thank You!, don't know. "
                # "Use three sentences maximum and keep the answer concise."
                # '''

                "You are an assistant for PDF question-answering tasks. "
                "Use the following pieces of retrieved context from uploaded PDF to answer "
                "the question. If you don't know the answer, say thank you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                '''
                <context>
                {context}
                <context>
                '''
            )

            # qa_system_prompt = (
            #     "You are a helpful assistant that provides concise and accurate answers "
            #     "based on the provided context. If the answer is not in the context, "
            #     "respond by saying 'Sorry, I don't have that information.'."
            # )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                return st.session_state['session_histories'][session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Chat Interface

            st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)
            st.subheader("💬 Chat with your Document")
            #st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)
            #user_input = st.text_input("Ask a question about your document:", key=f"user_input_{session_id}")
            
            user_input = st.chat_input("Ask DocuGenie....", key=f"user_input_{session_id}")


            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}},
                        )

                        if 'answer' in response:
                            # Update session history
                            st.session_state['session_histories'][session_id].add_user_message(user_input)
                            st.session_state['session_histories'][session_id].add_ai_message(response['answer'])

                            # Update responses and inputs
                            st.session_state['session_user_inputs'][session_id].append(user_input)
                            st.session_state['session_responses'][session_id].append(response['answer'])

                            # Display response
                            st.markdown(f"**You:** {user_input}")
                            st.markdown(f"**DocuGenie:** {response['answer']}")

                        else:
                            st.error("Unexpected response format from the model.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)

            # Display Conversation History
            if st.session_state['session_user_inputs'][session_id]:
                st.subheader("📝 Conversation History")
                for i in range(len(st.session_state['session_user_inputs'][session_id]) -1, -1, -1):
                    with st.expander(f"Q: {st.session_state['session_user_inputs'][session_id][i]}", expanded=False):
                        st.write(f"A: {st.session_state['session_responses'][session_id][i]}")

                st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)
        else:
            st.info("Please upload a PDF document and start interacting.")
else:
    st.info("**Please create or select a session from the sidebar to begin.**")


# CSS for overall page styling and chat message styling
css2 =  """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .chat-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
            background-color: #f9f9f9;
        }
        .user-message {
            background-color: #d1e7dd;
            text-align: left;
        }
        .ai-message {
            background-color: #f8d7da;
            text-align: left;
        }
        .stSidebar {
            background-color: #ffffff;
            border-right: 1px solid #ddd;
        }
        .st-expander {
            background-color: #f0f2f6;
        }
    </style>
    """,
#st.markdown(css2, unsafe_allow_html=True)
