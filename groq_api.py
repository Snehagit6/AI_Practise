#! /usr/bin/env python


import os

from dotenv import load_dotenv

from typing import List, Dict
import requests
from groq import Groq

response = None


load_dotenv()
api_key = os.getenv("GROK_API_KEY")
# client = Groq(api_key=api_key)
headers = {
    "Authorization": f"{api_key}",
    "Content-Type": "application/json"
}
messages = [
    {"role": "system", "content": "You are an expert software engineer that prefers functional programming."},
    {"role": "user", "content": "Write a function to swap the keys and values in a dictionary."}
]
url = "https://api.groq.com/openai/v1/chat/completions"


def generate_llm_api_response(model: str, messages: List[Dict]) -> str:
    """
    This function handles our LLM interactions using ChatML format.
    Call LLM to get response
    """
    try:
        payload = {"model": model, "messages": messages}
        response = requests.post(url=url, headers=headers, json=payload)
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     temperature=0.3,
        #     max_completion_tokens=512,
        #     top_p=1,
        #     reasoning_effort="medium",
        #     stream=False
        # )
        response.raise_for_status()
        print(f"Response content: {response.json()}")
    except Exception as e:
        print(f"Exception raised: {e}")
        raise
    finally:
        return response.json()

if __name__ == '__main__':
    generate_llm_api_response(model="meta-llama/llama-4-scout-17b-16e-instruct", messages=messages)