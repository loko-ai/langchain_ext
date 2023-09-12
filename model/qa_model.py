import json
from functools import lru_cache

from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from model.custom_retriever import CustomRetriever


@lru_cache(maxsize=1)
def get_embedding_function(collection):
    emb_args = json.loads(collection._collection.metadata['embeddings'])
    return OpenAIEmbeddings(**emb_args)


class OpenAIQAModel:
    def __init__(self, llm, chain_type, collection, n_sources, retriever_type, score_threshold, prompt_template):
        collection._embedding_function = get_embedding_function(collection)
        # self.chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
        #                                                          chain_type=chain_type,
        #                                                          retriever=collection.as_retriever(search_kwargs={"k": n_sources}),
        #                                                          return_source_documents=True)
        search_kwargs = {'k': n_sources, 'score_threshold': score_threshold}

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        chain_type_kwargs = {"prompt": PROMPT}
        self.chain = RetrievalQA.from_chain_type(llm=llm,
                                                 chain_type=chain_type,
                                                 retriever=CustomRetriever(vectorstore=collection,
                                                                           search_kwargs=search_kwargs,
                                                                           search_type=retriever_type),
                                                 chain_type_kwargs=chain_type_kwargs,
                                                 return_source_documents=True)

    def __call__(self, text):
        return self.chain({"query": text})