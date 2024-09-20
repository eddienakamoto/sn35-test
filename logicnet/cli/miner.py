from flask import Flask, request, jsonify
import openai
import os
import json
import asyncio

app = Flask(__name__)

openai_client = openai.AsyncOpenAI(
    # base_url=os.getenv("VLLM_API_URL"),
    base_url="http://157.157.221.29:30481/v1",
    api_key='xyz',
)

# openai_client = openai.AsyncOpenAI(
#     # base_url=os.getenv("VLLM_API_URL"),
#     base_url="http://66.114.112.70:40569/v1",
#     api_key='xyz',
# )

model = "Qwen/Qwen2-7B-Instruct"
# model = "edwardnakamoto/sn35-qwen-v1"


@app.route('/', methods=['POST'])
def handle_question():
    data = request.get_json()
    if 'question' in data:
        question = data['question']
        resp = asyncio.run(solve(question))
        return resp, 200

    return {"error": "Invalid request"}, 400


async def solve(question: str) -> dict[str, str]:

    resp = {"logic_answer": "", "logic_reasoning": ""}

    try:
        logic_question: str = question
        messages = [
            {"role": "user", "content": logic_question},
        ]
        print("making reqeust")
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
        )

        print("response received")
        resp['logic_reasoning'] = response.choices[0].message.content

        messages.extend(
            [
                {"role": "assistant", "content": resp['logic_reasoning']},
                {
                    "role": "user",
                    "content": "Give me the final short answer as a sentence. Don't reasoning anymore, just say the final answer in math latex.",
                },
            ]
        )

        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        resp["logic_answer"] = response.choices[0].message.content

        print(f"Response: {resp}")

        return resp
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # app.run(host='0.0.0.0', port=5001)
