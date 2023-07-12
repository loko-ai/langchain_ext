import json
from functools import lru_cache

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_embedding_function(collection):
    emb_args = json.loads(collection._collection.get(limit=1, include=['metadatas'])['metadatas'][0]['embeddings'])
    return OpenAIEmbeddings(**emb_args)


class OpenAIQAModel:
    def __init__(self, llm, chain_type, collection, n_sources):
        collection._embedding_function = get_embedding_function(collection)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                                 chain_type=chain_type,
                                                                 retriever=collection.as_retriever(search_kwargs={"k": n_sources}),
                                                                 return_source_documents=True)

    def __call__(self, text):
        return self.chain({"question": text})