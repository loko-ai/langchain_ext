import inspect
from typing import List, Any, Dict, Optional

import numpy as np
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.utils import maximal_marginal_relevance
from loguru import logger

from dao.chroma_dao import ChromaCollection

class CustomRetrievalQA(RetrievalQA):
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        retr_question = inputs['rquery']
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(retr_question, run_manager=_run_manager)
        else:
            docs = self._get_docs(retr_question)  # type: ignore[call-arg]
        logger.debug(f'CHAIN: {self.combine_documents_chain.__dict__}')
        logger.debug(f'CHAIN QUESTION: {question}')
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


class CustomRetriever(VectorStoreRetriever):
    vectorstore: ChromaCollection

    def get_relevant_documents(self, query: str) -> List[Document]:
        logger.debug(f'Retriever question: {query}')
        ### query ###
        k = self.search_kwargs.get('k', 4)
        n_res = k if self.search_type!='mmr' else 20
        include = ["metadatas", "documents", "distances"]
        if self.search_type=='mmr':
            n_res = self.search_kwargs.get('fetch_k', 20)
            include.append('embeddings')
        query_embedding = self.vectorstore._embedding_function.embed_query(query)
        res = self.vectorstore._query(query_embeddings=[query_embedding], n_results=n_res, include=include)
        # for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        #     print('score:', dist, 'metadata:', meta)


        if self.search_type == "similarity":
            res = self.vectorstore._results_to_docs_and_scores(res)
            docs = [doc for doc,score in res]
        elif self.search_type == "similarity_score_threshold":
            # print('SIMILARITY WITH THRESHOLD')
            score_threshold = self.search_kwargs.get('score_threshold')
            relevance_score_fn = self.vectorstore._select_relevance_score_fn()
            # print('score:', relevance_score_fn.__name__)
            res = self.vectorstore._results_to_docs_and_scores(res)
            # for doc, score in res:
            #     print('score:', score, 'new score:', relevance_score_fn(score), 'metadata:', doc.metadata)
            docs = [doc for doc, score in res if relevance_score_fn(score) > score_threshold]
        elif self.search_type == "mmr":
            # print('MMR')
            lambda_mult = self.search_kwargs.get('lambda_mult', .5)
            mmr_selected = maximal_marginal_relevance(
                np.array(query_embedding, dtype=np.float32),
                res["embeddings"][0],
                k=k,
                lambda_mult=lambda_mult,
            )
            res = self.vectorstore._results_to_docs_and_scores(res)
            # docs = [r[0] for i, r in enumerate(res) if i in mmr_selected]
            ### !!! ### [0, 13, 15, 2]
            # print(mmr_selected)
            docs = [(r[1],r[0]) for i, r in enumerate(res) if i in mmr_selected]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        if not docs:
            raise Exception("No documents retrieved")
        return docs


if __name__ == '__main__':
    from dao.chroma_dao import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    import os

    os.environ['OPENAI_API_KEY'] = ""

    class MockEmb:
        def embed_query(self, query):
            logger.debug('Compute mock emb')
            return emb

    collection_name = 'legal-500tk-nocontrib'
    chroma = Chroma(api_url='0.0.0.0:32773')
    print(chroma.list_collections())
    docsearch = chroma.create_collection(name=collection_name)
    include = ["metadatas", "documents", "embeddings"]
    emb = docsearch._get(include=include)["embeddings"][0]
    doc = docsearch._get(include=include)["documents"][0]
    # print(doc)
    docsearch._embedding_function = MockEmb()
    # docsearch._embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002')
    search_kwargs = {'k': 12, 'score_threshold': .85} # [0, 13, 15, 2, 6, 7, 8, 3, 11, 17, 9, 14]
    retriever_type = 'similarity' # similarity similarity_score_threshold mmr
    cr = CustomRetriever(vectorstore=docsearch, search_kwargs=search_kwargs, search_type=retriever_type)
    query = 'a quali tipi di imposta è soggetta la cessione di crediti in denaro? spiega in modo dettagliato citando le fonti'
    res = cr.get_relevant_documents(query)
    print('### result ###')
    for doc in res:
        print(doc)