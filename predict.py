import json
import re

from openai import OpenAI
from retrieval import search_wiki, search_kg


def call_api(model_name, prompt_system, prompt_user, max_tokens=24, api_base="http://localhost:8000/v1"):
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base,
    )
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        max_tokens=max_tokens
    )
    return chat_response.choices[0].message.content


def api_dicider(model_name, prompt_system, prompt_user, max_tokens=24):
    return call_api(model_name, prompt_system, prompt_user, max_tokens, api_base="http://localhost:8000/v1")


def api_rewriter(model_name, prompt_system, prompt_user, max_tokens=24):
    return call_api(model_name, prompt_system, prompt_user, max_tokens, api_base="http://localhost:8001/v1")


def api_responser(model_name, prompt_system, prompt_user, max_tokens=24):
    return call_api(model_name, prompt_system, prompt_user, max_tokens, api_base="http://localhost:8002/v1")


def match_subquestion(text):
    match = re.search(r"Subquestion:\s*(.+)", text)
    if match:
        question = match.group(1)
        return question
    else:
        return ""


def match_query(text):
    match = re.search(r"Query:\s*(.+)", text)
    if match:
        question = match.group(1)
        return question.strip()
    else:
        return ""


def iter_answer(question):
    """
    迭代推理
    :param question:
    :return:
    """
    iter_time = 0
    interact_history = question
    answer = question

    while iter_time <= 3:
        # 决策器
        dicider_input = interact_history
        dicider_prompt_system = "You are an expert in the field of natural language processing.You first need to determine whether you have enough information to answer the question; if you have enough information, you can reason out the answer directly; if you don't have enough information, you can rewrite the given question as a sub-question."
        dicider_prompt_user = dicider_input
        dicider_output = api_dicider('dicider', prompt_system=dicider_prompt_system, prompt_user=dicider_prompt_user)
        if "Final answer:" in dicider_output:
            answer += "\n" + dicider_output
            break

        # 重写器
        rewriter_prompt_system = "You are an expert in the field of natural language processing.You need to rewrite the sub-question as a refined query containing only the key question information."
        rewriter_prompt_user = dicider_output
        rewriter_output = api_rewriter('rewriter', prompt_system=rewriter_prompt_system,
                                       prompt_user=rewriter_prompt_user)

        # 应答器
        cur_query = match_query(rewriter_output)
        kg_result = search_kg(cur_query)
        wiki_result = search_wiki(cur_query)
        responser_prompt_system = "You are an expert in the field of natural language processing.You need to answer the sub-questions based on the retrieved information."
        responser_prompt_user = "Retrieved information:" + wiki_result + '\n' + kg_result + '\n' + dicider_output
        responser_output = api_responser('responser', prompt_system=responser_prompt_system,
                                         prompt_user=responser_prompt_user)

        interact_history += '\n' + dicider_output + '\n' + responser_output
        answer += '\n' + dicider_output + '\n' + rewriter_output + '\n' + responser_output
        iter_time += 1
    return answer


if __name__ == '__main__':
    print(iter_answer('Question: Where is the brokerage firm founded by the author of \"The Pursuit of Happyness\" headquartered?'))
