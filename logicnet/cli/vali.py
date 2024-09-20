from logicnet.challenger import LogicChallenger
from logicnet.rewarder import LogicRewarder
import logicnet as ln
import os
import requests
import concurrent.futures

urls = [
    "http://localhost:5000/",
    # "http://localhost:5001/"
]


def fetch(url, data):
    response = requests.post(url, json=data)
    return response.json()


def printd(model: str, reason: str, answer: str, score: float) -> None:
    print(f"Model: {model}")
    print(f"Reason: {reason}")
    print(f"Answer: {answer}")
    print(f"Score: {score}\n")
    print("******************************************************************")


if __name__ == "__main__":
    # Continuously call the server
    while True:
        challenger = LogicChallenger(
            base_url="http://213.173.110.25:17697/v1", api_key="xyz", model="Qwen/Qwen2-7B-Instruct"
        )

        rewarder = LogicRewarder(
            base_url="http://213.173.110.25:17697/v1", api_key="xyz", model="Qwen/Qwen2-7B-Instruct"
        )
        synapse_type = ln.protocol.LogicSynapse
        base_synapse = synapse_type(
            category="Logic", timeout=64
        )
        base_synapse = challenger(
            base_synapse
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch, urls[0], {"question": base_synapse.logic_question}),
                       #    executor.submit(fetch, urls[1], {
                       #                    "question": base_synapse.logic_question})
                       ]
            results = [future.result()
                       for future in concurrent.futures.as_completed(futures)]

        base_model_synapse = base_synapse.copy().miner_synapse()
        base_model_synapse.logic_reasoning = results[0]["logic_reasoning"]
        base_model_synapse.logic_answer = results[0]["logic_answer"]
        base_model_synapse.dendrite.process_time = 0.01
        base_model_synapse.dendrite.status_code = 200

        # my_model_synapse = base_synapse.copy().miner_synapse()
        # my_model_synapse.logic_reasoning = results[1]["logic_reasoning"]
        # my_model_synapse.logic_answer = results[1]["logic_answer"]
        # my_model_synapse.dendrite.process_time = 0.01
        # my_model_synapse.dendrite.status_code = 200

        # total_uids, rewards, reward_logs = rewarder(
        #     [1, 2], [base_model_synapse, my_model_synapse], base_synapse=base_synapse)

        total_uids, rewards, reward_logs = rewarder(
            [1], [base_model_synapse], base_synapse=base_synapse)

        printd("Base Model", base_model_synapse.logic_reasoning,
               base_model_synapse.logic_answer, rewards[0])
        # printd("My Model", my_model_synapse.logic_reasoning,
        #        my_model_synapse.logic_answer, rewards[1])

        # print(f"Base: {rewards[0]} --- My: {rewards[1]}")
