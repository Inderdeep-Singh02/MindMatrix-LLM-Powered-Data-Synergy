import re
from typing import List, Optional
# import aiofiles
import os 
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_SETTINGS = {
    "temperature": 0.8,
    "max_tokens": 150,
}

REFUSAL_PHRASES: List[str] = [
    "language model",
    "OpenAI",
    "not allowed",
    "not permitted",
    "not able to",
    # More phrases can be added in future after testing
]

ANSWER_KEY = ">ANS:"
ANSWER_KEY_PATTERN = ">ANS:"

class BaseAgent:
    verbose: bool
    openai_chat_model: BaseChatModel
    openrouter_chat_model: BaseChatModel

    def _init_(self, verbose: bool = False, **kwargs):
        self.verbose = verbose

    def load_openrouter_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("LLM_KEY"),
            openai_api_base=os.getenv("BASE_URL"),
            **DEFAULT_MODEL_SETTINGS if not kwargs else kwargs,
        )

    # def load_openai_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
    #     return ChatOpenAI(
    #         model_name=model_name,
    #         openai_api_key=settings.OPENAI_API_KEY,
    #         default_headers=settings.HELICONE_OPENAI_HEADERS,
    #         openai_api_base=settings.HELICONE_OPENAI_API_BASE,
    #         **DEFAULT_MODEL_SETTINGS if not kwargs else kwargs,
    #     )


    def parse_reflection_response(self, response: str, ANSWER_KEY_PATTERN: str) -> str:
        """Parses the response from the model to find the answer."""
        answer = ""
        match = re.search(f"{ANSWER_KEY_PATTERN}(.*)", response, re.IGNORECASE)
        if match:
            # Extract the matched group, which is the text following "ans:"
            answer_text = match.group(1)
            # Clean the extracted text by removing quotes
            cleaned_text = answer_text.replace('"', "")
            cleaned_text = cleaned_text.strip()
            answer = cleaned_text
        return answer

    def check_refusal_response(self, response: str) -> bool:
        """Checks if the response is a refusal to answer."""
        return any(re.search(phrase, response, re.IGNORECASE) for phrase in REFUSAL_PHRASES)


    async def load_prompt_template(self, prompt: str, input_variables: list[str]) -> PromptTemplate:
        return PromptTemplate(
            template=prompt,
            input_variables=input_variables,
        )

    # region Run ChatModel methods
    async def retry_run_chat_model(
        self, chat_model: BaseChatModel, chatml_inputs: List[BaseMessage], use_reflection = False
    ) -> str:
        """Retry invoking a chat model on either two failures: exception or refusal to answer"""
        retries = 0
        while retries < 3:
            try:
                response = await chat_model.ainvoke(chatml_inputs)
                response = response.content
                # More strict checking can be done from feedback responses
                if self.check_refusal_response(response):
                    retries += 1
                elif use_reflection:
                    response = self.parse_reflection_response(response, ANSWER_KEY_PATTERN)
                    if not response:
                        retries += 1
                    else:
                        return response
                else:
                    return response
            except Exception as e:
                print(str(e))
                retries += 1
        # If failed after retries, return an empty string
        return ""

    async def run_chat_model_instruct(
        self,
        chat_model: BaseChatModel,
        prompt_template: PromptTemplate,
        input_variables: dict,
        # ANSWER_KEY_PATTERN: str,
        use_reflection: Optional[bool] = False,
    ) -> str:
        """Runs a ChatModel in 'instruct' mode. This mode is just one user message (prompt)."""
        prompt = prompt_template.format(**input_variables)
        chatml_inputs = [SystemMessage(content=prompt)]
        response = await self.retry_run_chat_model(chat_model, chatml_inputs, use_reflection)
        return response

    