# Transposer-Model
# Transposer 🧠✨

**Transposition-Enhanced Representation Learning**  
*A lightweight architecture beyond attention, created by Abd, powered by matrix algebra and semantic intuition.*

---

## 🌟 What is Transposer?

Transposer is a novel neural architecture that learns semantic relationships between words **without any training**, **without GPUs**, and using **only NumPy**. It was inspired by matrix transposition from a 9th-grade mathematics textbook and runs in just ~23 seconds on a 2GB RAM machine.

> “While others require 100 billion tokens, Transposer generalizes from 3 lines.”

---

## 🔍 Features

- ✅ No attention, no transformers
- ✅ No gradient descent or training
- ✅ Works on CPUs with <2 GB RAM
- ✅ Generates semantic similarity from small datasets
- ✅ Built from first principles (matrix algebra, ReLU, transposition)
- ✅ Visualizes embeddings and relationships

---

## 🛠️ Requirements

Install the Python libraries with:

```bash
pip install -r requirements.txt


---

🧪 Running the Model

Make sure you have:

transposer.py (main code file)

data.txt (your text corpus, example provided)


Then run:

python transposer.py

It will:

Load the dataset from data.txt

Generate embeddings

Run the Transposer layer

Print semantically related words

Plot heatmaps and cosine similarity matrix



---

📄 Sample Output

--- Language Understanding Test ---

Related words to 'individuals': ['respiratory', 'greed', 'disrupting']
Related words to 'corporations': ['future', 'activities', 'adopt']


---

📚 Inspiration

> The idea was sparked while reading a transposition example in a 9th-grade math textbook.
Abd imagined: “What if instead of numbers, we transpose word embeddings?”
That idea became Transposer.




---

🔓 License

MIT — open source, free to use, modify, and share.


---

👋 Author

Built by Abd, a self-taught AI engineer, on a Phenom™ CPU with no GPU and a dream to break the myth that AI requires millions of dollars and massive compute.
