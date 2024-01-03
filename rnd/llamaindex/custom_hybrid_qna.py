import openai
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    StorageContext,
    load_index_from_storage,
    Prompt,
    download_loader,
    ServiceContext
)
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.retrievers import BM25Retriever
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine

from llama_index.llms import OpenAI

import os
import streamlit as st
import dotenv

st.set_page_config(page_title="FinDocChat", page_icon=':shark:')

# IGNORE THIS
######################################################################################3
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
    unsafe_allow_html=True)

st.write(
    f"""
       <div style="display: flex; align-items: center; margin-left: 0;">
           <h1 style="display: inline-block;">FinDoc'er</h1>
           <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
       </div>
       """,
    unsafe_allow_html=True,
)

###############################################################################


dotenv.load_dotenv('../../.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

data_path = "../../data/TRIG10.pdf"

USE_UNSTRUCTURED = False

filename = data_path.split("/")[-1].split('.')[0]

st.warning(
    f"Dealing with {filename} pdf here. Upload feature is not added into llamaindex demo yet. It works for langchain only. ")

template = (
    "We have provided context information below from a financial pdf document. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}. \n"
)

qa_template = Prompt(template)


@st.cache_data
def get_data():
    if USE_UNSTRUCTURED:
        UnstructuredReader = download_loader('UnstructuredReader')

        reader = SimpleDirectoryReader(input_files=[data_path], file_extractor={".pdf": UnstructuredReader()})
    else:
        # TODO should read from uploaded; just like how it's being done for langchain demo.
        reader = SimpleDirectoryReader(input_files=[data_path])
    data = reader.load_data()
    return data


def folder_available(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    print(folder_path, os.path.exists(folder_path))
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


@st.cache_resource
def get_vector_index():
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4")
    )
    filename = data_path.split("/")[-1].split('.')[0]
    if USE_UNSTRUCTURED:
        filename += "_UNSTRUCTURED"

    if folder_available(filename):
        st.write("Retrieving already stored index...")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=filename))
    else:
        st.write("Creating index...")

        index = VectorStoreIndex.from_documents(get_data(), service_context=service_context)
        index.storage_context.persist(f'./{filename}')

    return index


def get_keyword_based_retriever(index):
    service_context = ServiceContext.from_service_context(index.service_context)
    node_parser = service_context.node_parser
    nodes = node_parser.get_nodes_from_documents(get_data())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    return bm25_retriever


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # whatever and how many retrievers u use, get their nodes and combine it.

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


@st.cache_resource
def get_hybrid_retriever():
    vector_index = get_vector_index()
    vector_based_retriever = vector_index.as_retriever()
    keyword_based_retriever = get_keyword_based_retriever(vector_index)

    hybrid_retriever = HybridRetriever(vector_based_retriever, keyword_based_retriever)
    return hybrid_retriever, vector_index


@st.cache_resource
def get_re_ranker():
    # from llama_index.postprocessor import SentenceTransformerRerank
    # reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

    cohere_rerank = CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=3)
    return cohere_rerank


def get_query_engine():
    retriever, index = get_hybrid_retriever()
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[get_re_ranker()],
        service_context=index.service_context,
        text_qa_template=qa_template)
    return query_engine


def get_response(question):
    query_engine = get_query_engine()

    response = query_engine.query(
        question +
        " page reference after each statement."
    )

    return response


def main():
    st.info("Ready to answer questions.")

    user_question = st.text_input("Enter your question:")
    if user_question:
        response = get_response(user_question)
        st.success(response)

        for node in response.source_nodes:
            text_fmt = node.node.get_content().strip().replace("\n", " ")
            st.write(f"Text:\t {text_fmt} ...")
            st.write(f"Metadata:\t {node.node.metadata}")
            st.write(f"Score:\t {node.score:.3f}")


if __name__ == '__main__':
    main()
