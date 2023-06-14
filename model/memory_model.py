import faiss

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings


class MemoryModels:
    def __init__(self):
        self.windows = ConversationBufferWindowMemory
        self.summaries = ConversationSummaryBufferMemory
        self.vectors = self._get_vector_store_memory

    def _get_vector_store_memory(self, embedding_size=1536, search_kwargs=1, **kwargs):
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=search_kwargs))
        return VectorStoreRetrieverMemory(retriever=retriever)

    def get(self, model_type, **kwargs):
        model = self.__dict__[model_type]
        if hasattr(model, '__fields__'):
            kwargs = {k: v for k, v in kwargs.items() if k in model.__fields__}
        return model(**kwargs)


mm = MemoryModels()


class OpenAIModelMemory:
    def __init__(self, llm, memory_type='windows', **kwargs):
        template = """You are a chatbot having a conversation with a human. If the chatbot does not know the answer to a 
        question, it truthfully says it does not know.

        Relevant pieces of previous conversation:
        {history}

        (You do not need to use these pieces of information if not relevant)

        Current conversation:
        Human: {human_input}
        Chatbot:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )

        memory = mm.get(memory_type, llm=llm, **kwargs)

        self.llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

    def __call__(self, text):
        return self.llm_chain.predict(human_input=text)
