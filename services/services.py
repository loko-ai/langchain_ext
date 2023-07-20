import json

import chromadb
from bs4 import BeautifulSoup
from chromadb import Settings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from werkzeug.exceptions import NotFound

from doc.doc import llm_doc, llm_summary_doc, html_doc, llm_parser_doc, llm_qa_doc
from flask import Flask, request, jsonify
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from loko_extensions.business.decorators import extract_value_args
from loko_extensions.model.components import Component, Arg, save_extensions, Input, Select, Dynamic, MKVField, \
    MultiKeyValue, AsyncSelect, Output

from model.memory_model import OpenAIModelMemory
from model.parser_model import OpenAIParserModel
from model.qa_model import OpenAIQAModel

app = Flask("")

models = ['ada', 'babbage', 'code-cushman-001', 'code-cushman-002', 'code-davinci-001', 'code-davinci-002', 'curie',
          'davinci', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314',
          'text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003']

chatopenai_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301']

c = Component("LLM", args=[Select(name="model_name", options=models, value="text-davinci-003",
                                  description="Model name to use."),
                           Arg(name="max_tokens", description="The maximum number of tokens to generate in the "
                                                              "completion. -1 returns as many tokens as possible "
                                                              "given the prompt and the models maximal context size.",
                               type="number", value=256),
                           Arg(name="temperature", description="How creative the model should be.",
                               type="number", value=1.0),
                           Arg(name="memory", type="boolean", value=False),
                           Dynamic(name="type", dynamicType="select", parent="memory",
                                   options=["windows", "summaries", "vectors"],
                                   condition="{parent}"),
                           Dynamic(name="windows_k", label="k", dynamicType="number", value=5, parent="type",
                                   description="Number of interactions to keep in memory.",
                                   condition="{parent}==='windows'"),
                           Dynamic(name="summaries_max_token_limit", label="max_token_limit", dynamicType="number",
                                   value=2000, parent="type", description="Number of tokens to keep in memory.",
                                   condition="{parent}==='summaries'"),
                           Dynamic(name="vectors_embedding_size", label="embedding_size", dynamicType="number",
                                   value=1536, parent="type", description="Embedding size.",
                                   condition="{parent}==='vectors'"),
                           Dynamic(name="vectors_search_kwargs", label="search_kwargs", dynamicType="number",
                                   value=1, parent="type",
                                   condition="{parent}==='vectors'")
                           ], configured=True,
              description=llm_doc
              )
html = Component("HTML2Text", inputs=[Input("input", service="html2text")], configured=True, description=html_doc)
summary = Component("LLM Summary", inputs=[Input("input", service="summary_service")],
                    args=[Arg(name="chunk_size", value=700, type="number"),
                          Arg(name="chunk_overlap", value=70, type="number")],
                    description=llm_summary_doc)

output_parser = Component("LLM Parser", inputs=[Input("input", service="parser")],
                          args=[Select(name="model_name", options=models, value="text-davinci-003",
                                       description="Model name to use."),
                                Arg(name="max_tokens", description="The maximum number of tokens to generate in the "
                                                                   "completion. -1 returns as many tokens as possible "
                                                                   "given the prompt and the models maximal context size.",
                                    type="number", value=256),
                                Arg(name="temperature", description="How creative the model should be.",
                                    type="number", value=1.0),
                                MultiKeyValue(name='model',
                                              fields=[MKVField(name='field_name', required=True),
                                                      MKVField(name='field_type', required=True),
                                                      MKVField(name='field_description', required=True)])],
                          configured=False,
                          description=llm_parser_doc)

chroma = Component("Chroma", inputs=[Input("save", service="chroma_save", to="save"),
                                     Input("delete", service="chroma_delete", to="delete")],
                   outputs=[Output("save"), Output("delete")],
                   args=[Arg(name='existing_collection', label='use existing collection', type='boolean', value=True),
                         Dynamic(name="collection_name", parent="existing_collection",
                                 condition="{parent}", dynamicType='asyncSelect',
                                 url='http://localhost:9999/routes/langchain_ext/chroma_collections'),
                         Dynamic(name="new_collection_name", parent="existing_collection",
                                 condition="!{parent}",
                                 url='http://localhost:9999/routes/langchain_ext/chroma_collections'),
                         Arg(name="chunk_size", value=700, type="number"),
                         Arg(name="chunk_overlap", value=70, type="number"),
                         Arg(name='embeddings_model', value='text-embedding-ada-002')
                         ])

qa = Component("LLM QA", inputs=[Input("input", service="qa")],
                          args=[AsyncSelect(name="collection_name",
                                            url='http://localhost:9999/routes/langchain_ext/chroma_collections',
                                            value="langchain"),
                                Select(name="model_name", options=models, value="text-davinci-003",
                                       description="Model name to use."),
                                Arg(name="max_tokens", description="The maximum number of tokens to generate in the "
                                                                   "completion. -1 returns as many tokens as possible "
                                                                   "given the prompt and the models maximal context size.",
                                    type="number", value=256),
                                Arg(name="temperature", description="How creative the model should be.",
                                    type="number", value=1.0),
                                Select(name="chain_type", options=["stuff", "map_reduce", "refine", "map-rerank"],
                                       value="stuff"),
                                Arg(name="n_sources", type="number", value=1),
                                Select(name="retriever_type",
                                       options=["similarity", "similarity_score_threshold", "mmr"],
                                       value="similarity"),
                                Dynamic(name="score_threshold", parent="retriever_type",
                                        condition="{parent}==='similarity_score_threshold'",
                                        dynamicType="number", value=.2)
                                ],
               configured=False,
               description=llm_qa_doc)

save_extensions([c, html, summary, output_parser, chroma, qa])

models = dict()

# chroma settings
client_settings = Settings(chroma_api_impl="rest", chroma_server_host="langchain_ext_chromadb",
                           chroma_server_http_port="8000")

def handle_invalid_usage(exception):
    e = str(exception)
    j = dict(error=e)
    response = jsonify(j)

    if isinstance(exception, NotFound):
        logger.error(NotFound)
        response.status_code = 404
        return response
    logger.exception(e)

    status_code = getattr(exception, "status_code", None) or 500
    response.status_code = status_code
    logger.debug(f'ERROR: {status_code} - {j}')

    return response

@app.route("/", methods=["POST"])
@extract_value_args(_request=request)
def respond(value, args):
    temperature = args.get("temperature", 1)
    model_name = args.get("model_name", "text-davinci-003")
    max_tokens = args.get("max_tokens", 256)
    memory = args.get("memory")
    logger.debug(value)
    llm = OpenAI(model_name=model_name, max_tokens=int(max_tokens), temperature=float(temperature))
    if memory:
        logger.debug(f'ARGS: {args}')
        memory_type = args.get('type')
        memory_args = {k.replace(f'{memory_type}_', ''): int(v) for k, v in args.items() if k.startswith(f'{memory_type}_')}
        logger.debug(f'MEMORY: {memory_type} - ARGS: {memory_args}')
        if not models.get('llm_memory'):
            models['llm_memory'] = OpenAIModelMemory(llm=llm, memory_type=memory_type, **memory_args)
        llm = models.get('llm_memory')
        logger.debug(f'MEMORY: {llm.llm_chain.memory}')
        logger.debug(f'PROMPT: {llm.llm_chain.prompt}')

    logger.debug(f'LLM : {llm}')
    resp = llm(value)
    logger.debug("Response")
    logger.debug(resp)
    return jsonify(resp)


@app.route("/html2text", methods=["POST"])
@extract_value_args(_request=request)
def html2text(value, args):
    temp = BeautifulSoup(value)

    return jsonify(temp.get_text())


@app.route("/summary_service", methods=["POST"])
@extract_value_args(_request=request)
def summary_service(value, args):
    chunk_size = int(args.get("chunk_size"))
    chunk_overlap = int(args.get("chunk_overlap"))
    llm = OpenAI(temperature=0.9)
    ds = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs = ds.split_documents([Document(page_content=value)])
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)
    resp = chain.run(docs)

    return jsonify(resp)

@app.route("/parser", methods=["POST"])
@extract_value_args(_request=request)
def parser(value, args):
    temperature = args.get("temperature", 1)
    model_name = args.get("model_name", "text-davinci-003")
    max_tokens = args.get("max_tokens", 256)
    parser_model = args.get("model")

    logger.debug(f'ARGS: {args}')

    llm = OpenAI(model_name=model_name, max_tokens=int(max_tokens), temperature=float(temperature))
    parser = OpenAIParserModel(llm=llm, fields=parser_model)


    resp = parser(value)
    logger.debug("Response")
    logger.debug(resp)

    return jsonify(resp.__dict__)


@app.route("/chroma_save", methods=["POST"])
@extract_value_args(_request=request)
def chroma_save(value, args):
    collection_name = args.get("collection_name") or args.get("new_collection_name")
    chunk_size = int(args.get('chunk_size', 700))
    chunk_overlap = int(args.get('chunk_overlap', 70))
    model = args.get('embeddings_model', 'text-embedding-ada-002')
    embeddings = OpenAIEmbeddings(model=model)
    coll = Chroma(collection_name=collection_name,
                  embedding_function=embeddings, client_settings=client_settings)
    if isinstance(value, str):
        value = [dict(text=value, metadata=dict(source='no source'))]
    texts = [doc['text'] for doc in value]
    embeddings_bp = dict(model=model)
    metadatas = [dict(source=json.dumps(doc['metadata']['source']), embeddings=json.dumps(embeddings_bp)) for doc in value]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len)
    splitted_docs = text_splitter.create_documents(texts=texts, metadatas=metadatas)
    logger.debug(f'SPLITS: {len(splitted_docs)}')
    coll.add_documents(splitted_docs)
    # coll.persist()

    return jsonify('OK')

@app.route("/chroma_delete", methods=["POST"])
@extract_value_args(_request=request)
def chroma_delete(value, args):
    collection_name = args.get("collection_name") or args.get("new_collection_name")
    coll = Chroma(collection_name=collection_name, client_settings=client_settings)
    coll.delete_collection()
    # coll.persist()

    return jsonify(f'{collection_name} deleted')
@app.route("/chroma_collections", methods=["GET"])
def collections():

    client = chromadb.Client(client_settings)

    return jsonify([coll.name for coll in client.list_collections()])



@app.route("/qa", methods=["POST"])
@extract_value_args(_request=request)
def qa(value, args):
    collection_name = args.get("collection_name")
    temperature = float(args.get("temperature", 1))
    model_name = args.get("model_name", "text-davinci-003")
    max_tokens = int(args.get("max_tokens", 256))
    chain_type = args.get('chain_type', 'stuff')
    n_sources = int(args.get('n_sources', 1))
    retriever_type = args.get('retriever_type', 'similarity')
    score_threshold = float(args.get('score_threshold', .2))



    logger.debug(f'ARGS: {args}')

    docsearch = Chroma(collection_name=collection_name, client_settings=client_settings)

    if model_name in chatopenai_models:
        llm = ChatOpenAI(model_name=model_name, max_tokens=max_tokens, temperature=temperature)
    else:
        llm = OpenAI(model_name=model_name, max_tokens=max_tokens, temperature=temperature)

    qa = OpenAIQAModel(llm, chain_type, docsearch, n_sources, retriever_type, score_threshold)

    result = qa(value)
    logger.debug(result)
    return jsonify(dict(answer=result['result'],
                        sources=[dict(page_content=s.page_content, source=json.loads(s.metadata['source']))
                                 for s in result['source_documents']]))


app.register_error_handler(Exception, handle_invalid_usage)
if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
