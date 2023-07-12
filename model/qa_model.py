import json
from functools import lru_cache

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_embedding_function(collection):
    emb_args = json.loads(collection._collection.get(limit=1, include=['metadatas'])['metadatas'][0]['embeddings'])
    return OpenAIEmbeddings(**emb_args)


class OpenAIQAModel:
    def __init__(self, llm, chain_type, collection, n_sources, max_tokens):
        collection._embedding_function = get_embedding_function(collection)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                                 chain_type=chain_type,
                                                                 retriever=collection.as_retriever(k=n_sources),
                                                                 return_source_documents=True,
                                                                 # reduce_k_below_max_tokens=True,
                                                                 max_tokens_limit=max_tokens)

    def __call__(self, text):
        return self.chain({"question": text})