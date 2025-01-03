import json


def process_response(response):
    response = response.lstrip(" ```json").rstrip("```").strip()
    try:
        data = json.loads(response)
        return response
    except:
        print(response)
        return None
