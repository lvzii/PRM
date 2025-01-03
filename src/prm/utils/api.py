from openai import OpenAI


class aLLM(object):
    def __init__(self, api_port):
        self.client = OpenAI(
            base_url=f"http://localhost:{api_port}/v1",
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
