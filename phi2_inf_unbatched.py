import torch
import re
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import BitsAndBytesConfig

# Set model repository and device
model_id = "microsoft/phi-2"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     llm_int8_threshold=6.0
# )

# Load Phi-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
    # quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load HellaSwag dataset
# You can also substitute with your own dataset
# dataset = load_dataset("hellaswag", split="validation[:5]")  # For quick demo, use first 100 examples
dataset = load_dataset("tau/commonsense_qa", split="validation")

# Run inference on each example of hellaswag
# for example in dataset:
#     prompt = example['ctx']  # The context scenario
#     # You can format prompt as needed for QA/chat or in instruction style
#     full_prompt = f"Instruct: {prompt}\nOutput:\n"

#     inputs = tokenizer(full_prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=50)
#     result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("==============================")
#     print("Prompt:", prompt)
#     print("Generated:", result)

# Run inference on each example of commonsenseqa
total = 0
correct = 0
for example in dataset:  # For demo, process 5 questions. for example in dataset.select(range(5)):
    question = example["question"]
    # choices = [c["text"] for c in example["choices"]["label"]]
    labels = example["choices"]["label"]  # Typically ['A','B','C','D','E']
    choices_texts = example["choices"]["text"]
    # Build labeled choices for the prompt
    formatted_choices = ""
    for label, text in zip(labels, choices_texts):
        formatted_choices += f"{label}. {text}\n"

    prompt = (
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}"
        f"Choose the most suitable option (A, B, C, D, or E) and explain your reasoning."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    result = tokenizer.decode(outputs[0][inputs["input_ids"][0].shape[0]:], skip_special_tokens=True)
    print("===================================")
    # print(prompt)
    # print("Model output:", result)

    # Evaluate the model on test set:
    match = re.search(r'\b([A-E])\b', result)
    predicted = match.group(1) if match else ""
    gold = example["answerKey"]

    if predicted == gold:
        correct += 1
    total += 1

    print("===================================")
    print(f"idx: {total}", f"pred: {predicted}", f"gold: {gold}")

accuracy = correct / total if total > 0 else 0.0
print(f"Accuracy on CommonsenseQA: {accuracy:.4f} ({correct}/{total})")
