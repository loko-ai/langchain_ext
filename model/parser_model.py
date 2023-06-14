from typing import Dict, List, Optional

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import create_model
from pydantic.fields import FieldInfo


class OpenAIParserModel:
    def __init__(self, llm, fields):
        cfields = {field['field_name']: (eval(field['field_type']), FieldInfo(description=field['field_description'])) for field in fields}
        model = create_model('Custom', **cfields)
        parser = PydanticOutputParser(pydantic_object=model)
        self.output_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    def __call__(self, text):
        return self.output_parser.parse(text)