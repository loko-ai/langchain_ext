<html><p><a href="https://loko-ai.com/" target="_blank" rel="noopener"> <img style="vertical-align: middle;" src="https://user-images.githubusercontent.com/30443495/196493267-c328669c-10af-4670-bbfa-e3029e7fb874.png" width="8%" align="left" /> </a></p>
<h1>LangChain</h1><br></html>

**LangChain** extension relies on the 
<a href="https://python.langchain.com/en/latest/index.html">LangChain</a> framework to integrate Loko projects with 
<a href="https://platform.openai.com/">OpenAI</a> models.

### LLM
The **LLM** component allows to interact with Large Language Models. Depending on the template you use to request the 
model you can implement many different tasks like text classification, code generation, mail generation. 
<p align="center"><img src="https://github.com/Cecinamo/ner/assets/30443495/5c858c61-aa0b-41ce-b071-54f5237553a2" width="80%" /></p>

Within the block you can set the **model** you want to use, the **maximum number of tokens** to generate in the 
completion and the model **temperature**, representing the randomness of the answers. The model can keep trace of the 
previous interactions, model **memory** is based on **windows**, **summaries** or **vectors**:
- The **WindowMemory** uses the last K interaction to predict the next completion;
- The **SummaryMemory** creates a summary of the conversation over time;
- The **VectorStoreMemory** stores interactions into vectors and queries the top-K documents when it is called.

### LLM Summary

The **LLM Summary** component summarizes text basing on LLM. 

<p align="center"><img src="https://github.com/Cecinamo/ner/assets/30443495/901b21cc-7f7c-411f-acc4-0828f9c815d1" width="80%" /></p>

Within the block you can set **chunk_size** and **chunk_overlap** which refer to the preprocessing of the input text to 
summarize.

### HTML2Text

The **HTML2Text** component accepts HTML as input and extracts the textual content. 
<p align="center"><img src="https://github.com/Cecinamo/ner/assets/30443495/a9f1d595-a315-4379-83b0-3f4d14a779d5" width="80%" /></p>

### LLM Parser

The **LLM Parser** component allows to parse text basing on LLM.
<p align="center"><img src="https://github.com/Cecinamo/ner/assets/30443495/e76dd84f-e46f-4697-b84f-ac822bfc9c2d" width="80%" /></p>

Within the block you can set the **model** you want to use, the **maximum number of tokens** to generate in the 
completion and the model **temperature**, representing the randomness of the answers.

**Model** parameter defines the output structure.

**Example**:

Considering the model defined in the previous figure, we obtain: 

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

### Chroma

The **Chroma** component allows to save and delete ChromaDB collections.

The stored documents are previously split into chunks and vectorized by the component.

<p align="center"><img src="https://github.com/Cecinamo/ner/assets/30443495/fd7ae18c-9094-4016-b46f-1eef9fd56f12" width="80%" /></p>

Within the block you can set the **collection_name**, the **chunk_size** and **chunk_overlap** of the document's splits 
and the **embedding_model** used to vectorize the document's chunks.

In order to store documents into a ChromaDB collection, the component requires a list of dictionaries. Each dictionary 
represents a single document containing the *text* of the document and its *metadata*.

Example:

```
docs = [dict(text=doc['text'], metadata=dict(source=dict(fname=doc['fname']) for doc in docs]
``` 

### LMM QA

The **LLM QA** component allows to interact with Large Language Models basing on provided documents.

<p align="center"><img src="https://github.com/loko-ai/langchain_ext/assets/30443495/8795f0c0-edd4-41fc-8586-894511e79c5c" width="80%" /></p>

Within the block you can set a **Chroma collection name** to retrieve the sources necessary for processing the query, 
the **LLM model** you want to use to answer, the **maximum number of tokens** to generate in the  completion, the 
model **temperature**, representing the randomness of the answers and the **prompt_template**, representing the template
used by the LLM.

Each model has a maximum context length, meaning that when you need a high **number of sources** to answer to the query, 
you'll need to split the prompt into batches. Using the **chain type** parameter you can choose one of the following
methods:
- The **stuff** method uses all sources in the prompt;
- The **map_reduce** method separates sources into batches, the final answer is based on the answers from each 
batch;
- The **refine** method separates sources into batches, it refines the answer going through all the batches.
- **map_rerank** method separates sources into batches and assigns scores to the answers, the final answer is based 
on the high-scored answer from each batch.

Finally, you can set the measurement used to retrieve the relevant sources, which is the **retriever type**. 


## Configuration

In the file *config.json* you can set the **OPENAI API KEY** and configure the **chromadb**:

```
{
  "main": {
    "environment": {
      "OPENAI_API_KEY": "<insert your OPENAI API KEY here>"
    }
  },
  "side_containers": {
    "chromadb": {
      "image": "lokoai/chromadb",
      "environment": {
        "ANONYMIZED_TELEMETRY": "False",
        "ALLOW_RESET": "True",
        "IS_PERSISTENT": "TRUE"
      },
      "volumes": [
        "/var/opt/loko/chromadb/chroma:/chroma/chroma"
      ],
      "ports": {
        "8000": null
      }
    }
  }
}
```