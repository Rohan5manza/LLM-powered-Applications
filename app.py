import streamlit as st 
from dotenv import load_dotenv #to enable our app to access our variables inside .env file(our APi secret keys)
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub




def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text    




def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks=text_splitter.split_text(text)
    return chunks




def get_vectorstore(text_chunks):
    
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory=ConversationBufferMemory(memory_key='chat_history',retur_message=True)
    
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon="books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    

    st.header("Chat with multiple PDFs:books:")
    user_question=st.text_input("Ask a question")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your pdfs here and click on Process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text=get_pdf_text(pdf_docs)
                

                
                text_chunks=get_text_chunks(raw_text)

                vectorstore=get_vectorstore(text_chunks)

                st.session_state.conversation=get_conversation_chain(vectorstore)
    
    







if __name__=='__main__':
    main()