





### DOESN'T WORK!!!! BUG WITH LlamaIndex IMPLEMENTATION




from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)
import openai
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    StorageContext,
    load_index_from_storage,
    Prompt,
    ServiceContext
)

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



dotenv.load_dotenv(r'C:\Users\Catalect\Documents\GitHub\fin-doc-chat\.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

data_path = r"C:\Users\Catalect\Documents\GitHub\fin-doc-chat\data\TRIG10.pdf"

filename = data_path.split("\\")[-1].split('.')[0]

st.warning(
    f"Dealing with {filename} pdf here. Upload feature is not added into llamaindex demo yet. It works for langchain only. ")

template = (
    "We have provided context information below from a financial pdf document. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = Prompt(template)


@st.cache_data
def get_data(uploaded_files):
    # TODO should read from uploaded; just like how it's being done for langchain demo.
    reader = SimpleDirectoryReader(input_files=[data_path])
    data = reader.load_data()
    return data


def folder_available(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


@st.cache_resource
def get_query_engine():
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4")
    )
    filename = data_path.split("\\")[-1].split('.')[0]

    if folder_available(filename):
        st.write("Retrieving already stored index...")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=filename))
    else:
        st.write("Creating index...")

        index = VectorStoreIndex.from_documents(get_data(filename), service_context=service_context)
        index.storage_context.persist(f'./{filename}')

    query_engine = index.as_query_engine(text_qa_template=qa_template, streaming=True, similarity_top_k=3)
    return query_engine



@st.cache_resource
def get_hyde():
    return HyDEQueryTransform(include_original=True)


def get_response(question):
    query_engine = get_query_engine()
    hyde = get_hyde()
    hyde_query_engine = TransformQueryEngine(query_engine, hyde)

    response = hyde_query_engine.query(
        question +
        " page reference after each statement.")

    query_bundle = hyde(question)
    st.write("Got ", len(query_bundle.custom_embedding_strs))
    st.write(query_bundle.embedding_strs)
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
