# Overview
This project is a self-learning experiment to explore how LLMs work, from training to serving. It involves building and training a lightweight Transformer model using PyTorch, alongside a comparison with a pretrained GPT-2 model.  Instead of using a high-performance inference engine like TGI (Text Generation Inference), this project serves the model via FastAPI for lightweight and efficient deployment. The goal is to understand key LLM components—tokenization, training, and inference—while keeping the implementation simple. 


## 🛠️ Setup and Installation

### **1️⃣ Install Dependencies**

Make sure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### **2️⃣ Train the Model**

To train the model from scratch:

```bash
python train.py
```

This will generate the `trained_transformer.pth` file.

### **3️⃣ Run the API Server**

Start the FastAPI to serve text generation:

```bash
python app.py
```

The API will run at `http://localhost:5000`.


## 🖥️ API Usage

Generate text based on a given prompt.

#### **📝 Request**

```json
{
  "prompt": "The future of AI is"
}
```

#### **📌 cURL Example**

```bash
curl -X POST "http://localhost:5000/generate/custom" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The future of AI is"}'
```

#### **✅ Response**

```json
{
  "generated_text": "The future of AI is is is is is is is is is is is is is is is is is is is is is"
}
```

```bash
curl -X POST "http://localhost:5000/generate/gpt2" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The future of AI is"}'
```

#### **✅ Response**

```json
{
  "generated_text": "The future of AI is just getting started. AI is the future of artificial intelligence; it is the future of learning in AI. It is the future of artificial intelligence."
}
```


## 🐳 Running with Docker

### **1️⃣ Build the Docker Image**

```bash
docker build -t tiny-llm .
```

### **2️⃣ Run the Container**

```bash
docker run -p 5000:5000 tiny-llm
```

## **Input and Output Dimensions**
| Component | Input Shape | Output Shape (with `hidden_dim=128`) |
|-----------|------------|----------------------------------|
| **Embedding Layer** | `(1, N)` (tokens) | `(1, N, 128)` |
| **Transformer Block** | `(1, N, 128)` | `(1, N, 128)` |
| **Final Output Layer** | `(1, N, 128)` | `(1, N, 50257)` |

## Hidden_dim=128

hidden_dim represents the size of the internal feature representation in the Transformer model.

A higher hidden_dim means more capacity to learn complex patterns, but increases computation time and memory usage.

hidden_dim=128 is a balance between performance and efficiency, making it suitable for small-scale learning experiments.


## 🎯 Future Improvements

- Train on a larger dataset.
- Improve text generation quality.
- Add support for fine-tuning on custom data.
- Optimize inference for better performance.


## 📜 License

MIT License © 2025 [Alvin Phang].  
Feel free to use, modify, and distribute it. See the [LICENSE](LICENSE) file for details.


