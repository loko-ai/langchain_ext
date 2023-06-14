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


## Configuration

In the file *config.json* you can set the **OPENAI API KEY**:

```
{
  "main": {
    "environment": {
      "OPENAI_API_KEY": "<insert your OPENAI API KEY here>"
    }
  }
}
```