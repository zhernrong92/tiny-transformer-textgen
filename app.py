import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import TinyTransformer, tokenizer as custom_tokenizer

app = Flask(__name__)

# Load your custom TinyTransformer model
custom_model = TinyTransformer(custom_tokenizer.vocab_size)
custom_model.load_state_dict(torch.load("trained_transformer.pth"))
custom_model.eval()

# Load the pre-trained GPT-2 model
gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

### **1️⃣ Custom TinyTransformer Model Endpoint**
def generate_custom_text(prompt, max_length=20):
    """Manually generate text using TinyTransformer."""
    input_ids = custom_tokenizer(prompt, return_tensors="pt")["input_ids"]

    for _ in range(max_length):
        output = custom_model(input_ids)  # Get logits
        next_token = output.argmax(dim=-1)[:, -1].unsqueeze(0)  # Select most probable next token
        input_ids = torch.cat([input_ids, next_token], dim=-1)  # Append predicted token

    return custom_tokenizer.decode(input_ids[0], skip_special_tokens=True)

@app.route("/generate/custom", methods=["POST"])
def generate_custom():
    data = request.json
    prompt = data.get("prompt", "")
    generated_text = generate_custom_text(prompt)
    return jsonify({"generated_text": generated_text})

### **2️⃣ Pretrained GPT-2 Model Endpoint**
@app.route("/generate/gpt2", methods=["POST"])
def generate_gpt2():
    data = request.json
    prompt = data.get("prompt", "")
    input_ids = gpt2_tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Generate text using GPT-2
    output = gpt2_model.generate(input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95)
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

# Run Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
