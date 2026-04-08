import requests
from trl import GRPOTrainer, GRPOConfig
from trl.chat_template_utils import parse_response, add_response_schema
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Tokenizer
from datasets import load_dataset
import os

from grpo_agent import em_check
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model = AutoModelForCausalLM.from_pretrained("/Users/likun/code/models/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("/Users/likun/code/models/Qwen3-0.6B")
add_response_schema(tokenizer)
# model.to("cuda")
model.eval()
def search(query: str):
    """
    Search for relevant documents based on the query using dense retrieval.

    Args:
        query: The search query.

    Returns:
        A string containing the formatted search results.
    """
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

def tool_executor(tool_call):
    # for tool_call in tool_calls:
    if tool_call["function"]["name"] == "search":
        query = tool_call["function"]["arguments"]["query"]
        output = search(query)
            # tool_call["output"] = output
    return output

def infer(query):
    count = 0
    answer = ""
    # flag = True
    messages = [
    {"role": "system", "content": "Answer the given question. Each time you obtain new information, you must think and reason. \
     After thinking, if you find that you lack certain knowledge, you can acquire it through tools and obtain relevant information. \
     You can search as many times as your want.If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\
     The following is the user's question."},
    {"role": "user", "content": f"call tool get answer of "+query}
    ]
    while count < 5 and not answer:
        inputs = tokenizer.apply_chat_template(messages, tools=[search], add_generation_prompt=True, return_dict=True, return_tensors="pt",enable_thinking=False)
        # print(tokenizer.batch_decode(inputs["input_ids"]))
        outputs = model.generate(**inputs.to(model.device), max_new_tokens=1280)
        # add_response_schema(tokenizer)
        # print(parse_response(tokenizer, outputs[0][len(inputs["input_ids"][0]):]))
        print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))

        output = parse_response(tokenizer, outputs[0][len(inputs["input_ids"][0]):])
        if "tool_calls" in output:
            print("工具调用结果：", output["tool_calls"][0])
            flag = True
            messages.append({"role": "assistant", "content": output["tool_calls"][0]})
            tool_res = tool_executor(output["tool_calls"][0])
            messages.append({"role": "tool", "content": tool_res})
        else:
            # print(output["content"])
            answer = output["content"]


        count += 1
    return answer

# answer = infer("Which film whose director is younger, Charge It To Me or Danger: Diabolik?")
# print(answer)
scores = []
dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="test")
for example in dataset.select(range(10)):
    # print(example)
    question = example["question"]
    answer = example["golden_answers"]
    print("Question: ", question)
    pred_answer = infer(question)
    print("Answer: ", pred_answer)
    score = em_check(pred_answer, answer)
    scores.append(score)
print("EM Score: ", sum(scores)/len(scores))