from openai import OpenAI
import os 
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("LLM_KEY")
model = os.getenv("MODEL")


async def prompt_llm(prompt_template):
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
    )

    completion = client.chat.completions.create(
      model="nousresearch/nous-capybara-7b:free",
      messages=[
        {
          "role": "user",
          "content": prompt_template,
        },
      ],
    )

    return completion.choices[0].message.content.strip()
