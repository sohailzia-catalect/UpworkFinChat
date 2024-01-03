

in the root dir, run the following:

`pip install -r reqs.txt`

Make sure you have the same versions of openai and llamaindex/langchain. 
To check version, you can use the command: 
`pip show <library_name>` 
For example: 
`pip show openai`
`pip show llama-index`

### Milestone 1 : 
In LlamaIndex dir > qna_basic.py and custom_hybrid_qna.py

### Milestone 2:

The langchain hyde works. 
The LlamaIndex based hyde implementation doesn't work, it's stuck in loop as mentioned in the last call as well. Seems like a bug in their code. 
So, the langchain based Hyde works. 

In Langchain dir > hyde_qna.py.

## OpenAI Assistant. 
I have also pushed extra work that I did. assistant_qna.py
It's basically letting openai do the RAG stuff on its own.
For more information: https://platform.openai.com/docs/assistants/overview





