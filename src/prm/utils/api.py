from openai import OpenAI


class aLLM(object):
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="0",
        )

    def call(self, instruction):
        messages = [{"role": "user", "content": instruction}]
        response = self.client.chat.completions.create(
            messages=messages,
            model="model",
        )
        response = response.choices[0].message.content
        return response
