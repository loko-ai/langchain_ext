llm_doc = '''### Description
The **LLM** component allows to interact with Large Language Models.

### Configuration

- **model_name** parameter sets the model you want to use.
- **max_tokens** represents the maximum number of tokens to generate in the completion. 1 returns as many tokens as 
possible given the prompt and the models maximal context size.
- **temperature** represents how creative the model should be. Lower values make the model more deterministic.
- **memory** parameter allows to keep trace of the previous interactions. Memory is based on **windows**, **summaries** 
or **vectors**:
    - The **WindowMemory** uses the last K interaction to predict the next completion; 
    - The **SummaryMemory** creates a summary of the conversation over time;
    - The **VectorStoreMemory** stores interactions into vectors and queries the top-K documents when it is called.
 
'''

llm_summary_doc = '''### Description
The **LLM Summary** uses the Large Language Models to implement summarization.

### Configuration

- **chunk_size** parameter sets the maximum number of characters of the single chunks used by the LLM.
- **chunk_overlap** represents the overlap between chunks. 
'''

html_doc = '''### Description
The **HTML2Text** extracts text from HTML. 
'''

llm_parser_doc = '''### Description
The **LLM Parser** allows to parse input using Large Language Models.

### Configuration

- **model_name** parameter sets the model you want to use.
- **max_tokens** represents the maximum number of tokens to generate in the completion. -1 returns as many tokens as 
possible given the prompt and the models maximal context size.
- **temperature** represents how creative the model should be. Lower values make the model more deterministic.
- **model** represents the fields structure you need as the output. Each field needs a **field_name**, **field_type**,
**field_description**.
 
**Example**:

Model:

```
<field_name>         <field_type>         <field_description>
name                 str                  name of an actor                       
film_names           List[str]            list of names of films they starred in 
```

Input: 
```
Tom Hanks acted in Forrest Gump and Apollo 13.
```

Output:
```
{
    "film_names": ["Forrest Gump","Apollo 13"],
    "name": "Tom Hanks"
}
```

'''

chroma_doc = '''### Description
The **Chroma** component allows to save and delete ChromaDB collections.

The stored documents are previously split into chunks and vectorized by the component. 

### Configuration

- **collection_name** sets the chroma collection name.
- **chunk_size** parameter sets the maximum number of characters of a single chunk (portion of a document).
- **chunk_overlap** represents the overlap between chunks. 
- **embeddings_model** represents the model used to vectorize the document's chunks.

### Input

In order to store documents into a ChromaDB collection, the component requires a list of dictionaries. Each dictionary 
represents a single document containing the *text* of the document and its *metadata*.

Example:

```
docs = [dict(text=doc['text'], metadata=dict(source=dict(fname=doc['fname']) for doc in docs]
``` 

'''


llm_qa_doc = '''### Description
The **LLM QA** component allows to interact with Large Language Models basing on provided documents.

### Configuration

- **collection_name** sets the chroma collection used to answer to the query.
- **model_name** parameter sets the model you want to use.
- **max_tokens** represents the maximum number of tokens to generate in the completion. 1 returns as many tokens as 
possible given the prompt and the models maximal context size.
- **temperature** represents how creative the model should be. Lower values make the model more deterministic.
- **chain_type** parameter:
    - The **stuff** method uses all sources in the prompt; 
    - The **map_reduce** method separates sources into batches, the final answer is based on the answers from each 
    batch;
    - The **refine** method separates sources into batches, it refines the answer going through all the batches.
    - **map_rerank** method separates sources into batches and assigns scores to the answers, the final answer is based 
    on the high-scored answer from each batch. 
- **n_sources** represents the number of sources used to answer to the query.
- **retriever_type** sets the measurement used to retrieve the relevant sources.

'''