import openai
import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
import os
import glob

st.set_page_config(page_title="FinDocChat", page_icon=':shark:')


openai.api_key = '<ADD YOU OWN>'

@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = file_path.type.split('/')[-1]
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == "txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide pdf.', icon="⚠️")
    st.write("alltext ", all_text)
    return all_text


def get_retriever(filename):
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.load_local(filename, embeddings)
    return retriever.as_retriever()


def folder_available(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    folder_path += "_index"
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


@st.cache_resource
def create_retriever(filename, _embeddings):
    ony_filename = filename[0].name.split('.')[0]
    if not folder_available(ony_filename):
        try:
            os.mkdir(r'C:\Users\Catalect\PycharmProjects\OthmanFinProject\RnD\{}_index'.format(ony_filename))
        except:
            print("Already created")

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(filename)

        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1500,
                             overlap=200)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        vectorstore = FAISS.from_texts(splits, _embeddings)
        retriever = vectorstore.as_retriever(k=5)
        vectorstore.save_local(r'C:\Users\Catalect\PycharmProjects\OthmanFinProject\RnD\{}_index'.format(ony_filename))
        return retriever

    else:
        return get_retriever(r'C:\Users\Catalect\PycharmProjects\OthmanFinProject\RnD\{}_index'.format(ony_filename))


@st.cache_resource
def split_texts(text, chunk_size, overlap):

    st.info("`Splitting doc ...`")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    st.write(text)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits


def main():
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)

    # Add custom CSS
    st.markdown(
        """
        <style>

        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }

            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }

            .css-zt5igj {left:0;
            }

            span.css-10trblm {margin-left:0;
            }

            div.css-1kyxreq {margin-top: -40px;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("img/logo1.png")

    st.write(
        f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">FinDoc'er</h1>
        <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Menu")

    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings", "HuggingFace Embeddings(slower)"])

    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH"])

    os.environ["OPENAI_API_KEY"] = '<ADD YOU OWN>'

    uploaded_files = st.file_uploader("Upload a your PDF", type=[
        "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files

        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        elif embedding_option == "HuggingFace Embeddings(slower)":
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(uploaded_files, embeddings)

        chat_openai = ChatOpenAI(model_name="gpt-4", verbose=True, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        st.info("Ready to answer questions.")

        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()