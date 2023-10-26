import sys
import uuid
from typing import List, Any, Tuple

import httpx
from langchain.schema import Document
from langchain.vectorstores import Chroma
from loguru import logger


class DocItem:
    def __init__(self, text, item_id=None, metadata=None, embeddings=None, emb_function=None):

        self.item_id = item_id or str(uuid.uuid1())
        self.text = text
        self.metadata = metadata if metadata is not None else {}
        self.embeddings = embeddings or emb_function.embed_query(text)

    def __str__(self):
        return f"ID: {self.item_id}\nText: {self.text}\nMetadata: {self.metadata}\nEmbeddings: {self.embeddings}"

class ChromaCollection(Chroma):

    def __init__(self, client, name, id, embedding_function=None, metadata=None, relevance_score_fn=None):
        self._client = client
        self._embedding_function = embedding_function
        self.name = name
        self.metadata = metadata
        self.id = id
        self.override_relevance_score_fn = relevance_score_fn
        self._collection = self

    def __repr__(self) -> str:
        return f"Collection(name={self.name})"

    def _upsert(self, ids, embeddings, metadatas=None, documents=None, increment_index=True) -> bool:
        """
        Upserts a batch of embeddings in the database
        - pass in column oriented data lists
        """
        url = self._client._api_url + "/collections/" + str(self.id) + "/upsert"
        data = {"ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "documents": documents,
                "increment_index": increment_index}
        resp = httpx.post(url, json=data)
        resp.raise_for_status()
        return True

    def _delete(self):
        httpx.delete(self._client._api_url + "/collections/" + self.name).raise_for_status()

    def _get(self, ids=None, where=None, sort=None, limit=10, offset=None,
             page=None, page_size=None, where_document=None, include=None):
        where = where or {}
        where_document = where_document or {}
        include = include or ["metadatas", "documents"]
        url = self._client._api_url + "/collections/" + str(self.id) + "/get"

        if page and page_size:
            offset = (page - 1) * page_size
            limit = page_size

        data = {"ids": ids,
                "where": where,
                "sort": sort,
                "limit": limit,
                "offset": offset,
                "where_document": where_document,
                "include": include}
        resp = httpx.post(url, json=data)
        resp.raise_for_status()

        return resp.json()

    def _query(self, query_embeddings, n_results: int = 10, where=None, where_document=None,
               include=None):

        where = where or {}
        where_document = where_document or {}
        include = include or ["metadatas", "documents", "distances"]
        """Gets the nearest neighbors of a single embedding"""
        url = self._client._api_url + "/collections/" + str(self.id) + "/query"
        data = {"query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
                "include": include}

        resp = httpx.post(url, json=data).raise_for_status()

        return resp.json()

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        return [
            # TODO: Chroma can do batch querying,
            # we shouldn't hard code to the 1st result
            (Document(page_content=result[0], metadata=result[1] or {}), result[2])
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


    def add_documents(self, docs, emb_function):

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for doc in docs:
            doc = DocItem(text=doc.page_content, metadata=doc.metadata, emb_function=emb_function)
            ids.append(doc.item_id)
            documents.append(doc.text)
            metadatas.append(doc.metadata)
            embeddings.append(doc.embeddings)

        self._upsert(metadatas=metadatas, embeddings=embeddings, documents=documents, ids=ids)

        return ids


class Chroma:

    def __init__(self, api_url):
        self._api_url = f"http://{api_url}/api/v1"

    def create_collection(self, name, metadata=None, embedding_function=None, get_or_create=True):
        url = self._api_url + "/collections"
        data = {"name": name, "metadata": metadata, "get_or_create": get_or_create}
        resp = httpx.post(url, json=data).raise_for_status()
        resp_json = resp.json()

        logger.debug(f'Collection: {resp_json}')

        return ChromaCollection(
            client=self,
            id=resp_json["id"],
            name=resp_json["name"],
            embedding_function=embedding_function,
            metadata=resp_json["metadata"],
        )

    def list_collections(self):
        res = (httpx.get(self._api_url+'/collections').json())
        return [ChromaCollection(self, **json_col) for json_col in res]

    def delete_collection(self, name: str) -> None:
        """Deletes a collection"""
        httpx.delete(self._api_url + "/collections/" + name).raise_for_status()





if __name__ == '__main__':

    chroma = Chroma(api_url='0.0.0.0:32776')
    # chroma.delete_collection('ciccio')

    print([col.__dict__ for col in chroma.list_collections()])

    coll = chroma.create_collection(name='coll_prova', get_or_create=True)
    print('qui', [col.__dict__ for col in chroma.list_collections()])

    # text_items = [Document(text='ciao ciao', emb_function=coll._embedding_function) for _ in range(10)]
    # chroma.add_texts(coll, text_items)

    res = coll._get(include=["metadatas", "documents", "embeddings"])

    # res = chroma.query(coll.id, [], 10)

    print(res)

    sys.exit(0)
    print(coll.save(ids=['1','2'], embeddings=[], metadatas=[], documents=['ciao', 'hello'], increment_index=False))

    client_settings = Settings(chroma_api_impl="rest", chroma_server_host="0.0.0.0",
                               chroma_server_http_port="32776")

    client = chromadb.Client(client_settings)

    print([col.__dict__ for col in client.list_collections()])
