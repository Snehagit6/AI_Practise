import os
# from litellm import completion
from openai import OpenAI
import litellm
from typing import List, Dict
# from google.colab import userdata
# api_key = userdata.get('OPENAI_API_KEY')
# litellm._turn_on_debug()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10
    )
    return response.choices[0].message.content


messages = [
    {"role": "system", "content": "You are an expert software engineer that prefers functional programming."},
    {"role": "user", "content": "Write a function to swap the keys and values in a dictionary."}
]

response = generate_response(messages)
print(response)



# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
#
# inputs = tokenizer("What is functional programming?", return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
