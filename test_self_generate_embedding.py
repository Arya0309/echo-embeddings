import transformers

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda"  # Change to "cpu" if you don't have a GPU

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)
model.eval()
model.to(device)

query = "She loves to travel in summer, especially to cold destinations, avoiding hot and crowded places"
template = "Rewrite the following sentence: {query}. The rewritten sentence:"

inputs = tokenizer(template.format(query=query), return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
generate_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9,
    temperature=0.6,
    repetition_penalty=1.2,
)
# print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0])

output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
rewritten = output.split("The rewritten sentence:")[-1].strip()
