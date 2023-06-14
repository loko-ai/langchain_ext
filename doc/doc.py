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