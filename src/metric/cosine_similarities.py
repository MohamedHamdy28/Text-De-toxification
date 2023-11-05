import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CosSimilarity:
    def __init__(self) -> None:
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def batch_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = self.model(**inputs)

        # Take the average of the hidden states to get sentence embeddings
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).detach().cpu().numpy()
        return embeddings
    
    def calculate_cosine_similarity(self, list1, list2, batch_size=32):
        total_similarities = []
        for i in range(0, len(list1), batch_size):
            # Process each batch
            batch1 = list1[i:i+batch_size]
            batch2 = list2[i:i+batch_size]

            embeddings1 = self.batch_embeddings(batch1)
            embeddings2 = self.batch_embeddings(batch2)

            # Calculate cosine similarity for the batch
            similarities = cosine_similarity(embeddings1, embeddings2).diagonal()
            total_similarities.extend(similarities)
        return total_similarities