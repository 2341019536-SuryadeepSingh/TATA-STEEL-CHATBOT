import streamlit as st
from dotenv import load_dotenv
import os
import io
from PIL import Image

# --- New/Updated Imports ---
# You'll need to install PyMuPDF: pip install PyMuPDF
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Your HTML templates (assuming htmlTemplates.py exists)
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def process_pdf_documents(pdf_docs):
    """
    Extracts text, page count, and images from a list of PDF documents.
    Returns:
        - A list of text chunks, each with its page number as metadata.
        - The total page count.
        - A list of dictionaries, where each dict contains image bytes and its original page number.
    """
    all_text_chunks_with_metadata = []
    page_count = 0
    images = []
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    for pdf_file in pdf_docs:
        # Open the PDF file with PyMuPDF
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        page_count += pdf_document.page_count
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Extract text from the page
            page_text = page.get_text()
            
            # Split the page text into chunks
            chunks = text_splitter.split_text(page_text)
            
            # Add metadata (page number) to each chunk
            for chunk in chunks:
                all_text_chunks_with_metadata.append({"text": chunk, "metadata": {"page": page_num}})

            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                images.append({"page": page_num, "bytes": image_bytes})
        
        pdf_document.close()
        
    return all_text_chunks_with_metadata, page_count, images


def get_vectorstore(chunks_with_metadata):
    """Creates a FAISS vector store from text chunks with metadata."""
    if not GEMINI_API_KEY:
        st.error("Google Gemini API key is not set. Please add it to your .env file as GEMINI_API_KEY.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        
        # Separate texts and metadatas for FAISS
        texts = [d['text'] for d in chunks_with_metadata]
        metadatas = [d['metadata'] for d in chunks_with_metadata]

        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store with Gemini. Error: {e}")
        return None

def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain using the Gemini chat model."""
    if not GEMINI_API_KEY:
        st.error("Google Gemini API key is not set.")
        return None
    try:
        prompt_template = """You are a helpful assistant for answering questions based on the provided document.
        Use the following pieces of context, which are excerpts from a PDF the user uploaded, to answer the question at the end.
        If the answer is not in the provided context, just say that you cannot find the answer in the document. Don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True, google_api_key=GEMINI_API_KEY)
        
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True, output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True # This is crucial for getting page numbers
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Failed to create conversation chain with Gemini. Error: {e}")
        return None

def handle_userinput(user_question):
    """Handles user input, displays the conversation, and shows relevant images."""
    response = None
    if "how many pages" in user_question.lower():
        if "page_count" in st.session_state and st.session_state.page_count > 0:
            page_count = st.session_state.page_count
            response_text = f"The uploaded document(s) have a total of {page_count} pages."
            st.session_state.chat_history.append({'role': 'user', 'content': user_question})
            st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
    elif st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if hasattr(message, 'type') and message.type == 'human':
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif hasattr(message, 'type') and message.type == 'ai':
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else: # Custom dict for page count
                role = message.get('role')
                content = message.get('content')
                if role == 'user':
                    st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)

    # Display source images if available
    if response and 'source_documents' in response:
        source_pages = {doc.metadata.get('page', -1) for doc in response['source_documents']}
        relevant_images = [img for img in st.session_state.images if img['page'] in source_pages]
        
        if relevant_images:
            st.write("### Relevant Images from the Document")
            for img_data in relevant_images:
                try:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                    st.image(img, caption=f"From page {img_data['page'] + 1}")
                except Exception as e:
                    st.warning(f"Could not display an image from page {img_data['page'] + 1}. It may be in an unsupported format.")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with multiple PDFs (Gemini)", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "page_count" not in st.session_state:
        st.session_state.page_count = 0
    if "images" not in st.session_state:
        st.session_state.images = []

    st.header("Chat with multiple PDFs using Gemini 📚")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation or "how many pages" in user_question.lower():
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process your documents before asking a question.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if not GEMINI_API_KEY:
                 st.error("Please set your GEMINI_API_KEY in the .env file first.")
            elif pdf_docs:
                with st.spinner("Processing documents with Gemini..."):
                    # 1. Process PDFs to get text chunks with metadata, page count, and images
                    chunks_with_metadata, page_count, images = process_pdf_documents(pdf_docs)
                    st.session_state.page_count = page_count
                    st.session_state.images = images
                    
                    if chunks_with_metadata:
                        # 2. Create vector store from the chunks with metadata
                        vectorstore = get_vectorstore(chunks_with_metadata)
                        
                        if vectorstore:
                            # 3. Create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.chat_history = []
                            st.success(f"Processing complete ({page_count} pages, {len(images)} images)! You can now ask questions.")
                    else:
                        st.warning("Could not extract any text from the uploaded PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
