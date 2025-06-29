import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset from file
with open("data.txt", encoding='utf-8') as f:
    text = f.read()

# 1. Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower().split()


tokens = preprocess_text(text)
vocab = sorted(set(tokens))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}  # Added for QA

token_ids = [word_to_idx[word] for word in tokens]
sequence_len = len(token_ids)
vocab_size = len(vocab)
embedding_dim = 64

# 2. Embedding matrix
np.random.seed(80)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)
input_embeddings = np.array([embedding_matrix[i] for i in token_ids])  # shape: (seq_len, embed_dim)

# 3. Positional Encoding
def positional_encoding(seq_len, embed_dim):
    pe = np.zeros((seq_len, embed_dim))
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/embed_dim)))
            if i+1 < embed_dim:
                pe[pos, i+1] = np.cos(pos / (10000 ** ((2 * i)/embed_dim)))
    return pe

pos_encoding = positional_encoding(sequence_len, embedding_dim)
input_embeddings += pos_encoding

# 4. Transposition Layer
transposed = input_embeddings.T  # shape: (embed_dim, seq_len)

# 5. Transformation Layer
W1 = np.random.randn(embedding_dim, embedding_dim)
W2 = np.random.randn(embedding_dim, embedding_dim)

hidden = np.maximum(0, W1 @ transposed)     # ReLU
transformed = W2 @ hidden                   # shape: (embed_dim, seq_len)
output_transposed = transformed.T          # shape: (seq_len, embed_dim)

# 6. Combine outputs
final_output = input_embeddings + output_transposed

# 7. Target embeddings (dummy target for learning signal)
target_embeddings = np.random.randn(sequence_len, embedding_dim)

# 8. Training loop
learning_rate = 0.0001
losses = []
epochs = 1000

for epoch in range(epochs):
    predictions = final_output
    loss = np.mean((predictions - target_embeddings)**2)
    losses.append(loss)

    grad_output = 2 * (predictions - target_embeddings) / sequence_len
    grad_output_T = grad_output.T
    grad_W2 = grad_output_T @ hidden.T
    grad_hidden = W2.T @ grad_output_T
    grad_hidden[hidden <= 0] = 0  # ReLU derivative
    grad_W1 = grad_hidden @ transposed.T

    W1 -= learning_rate * grad_W1
    W2 -= learning_rate * grad_W2

    # Recompute layers
    hidden = np.maximum(0, W1 @ transposed)
    transformed = W2 @ hidden
    output_transposed = transformed.T
    final_output = input_embeddings + output_transposed

# 9. Visualization
def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap='viridis')
    plt.title(title)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Token Index")
    plt.tight_layout()
    plt.show()

plot_heatmap(input_embeddings, "Raw + Positional Embeddings")
plot_heatmap(output_transposed, "Transposed Embeddings After Training")
plot_heatmap(final_output, "Final Output Embeddings After Training")

# 10. Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Curve Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Cosine Similarity Matrix (Language Understanding)
def cosine_similarity_matrix(vectors):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norm + 1e-8)
    similarity = normalized @ normalized.T
    return similarity

sim_matrix = cosine_similarity_matrix(final_output)
plot_heatmap(sim_matrix, "Cosine Similarity Matrix (Final Output)")

# 12. Basic QA: Predict most similar words
def predict_related_words(word, top_n=3):
    if word not in word_to_idx:
        return f"Word '{word}' not in vocabulary."
    
    idx = word_to_idx[word]
    embedding = final_output[idx]  # get embedding of that word
    sims = cosine_similarity(embedding.reshape(1, -1), final_output)[0]
    
    # Get top N indices (excluding the word itself)
    top_indices = sims.argsort()[-top_n-1:][::-1]
    top_indices = [i for i in top_indices if i != idx][:top_n]
    
    return [tokens[i] for i in top_indices]

# --- Testing the language model ---
print("\n--- Language Understanding Test ---")
print("Related words to 'contamination':", predict_related_words('contamination'))
print("Related words to 'human':", predict_related_words('human'))
print("Related words to 'individuals':", predict_related_words('individuals'))
print("Related words to 'corporations':", predict_related_words('corporations'))
print("Related words to 'population':", predict_related_words('population'))
