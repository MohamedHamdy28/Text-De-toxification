import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CosSimilarity:
    def __init__(self) -> None:
        """
        The initializer for the CosSimilarity class.
        
        Attributes:
            device (torch.device): Device configuration to use GPU if available.
            tokenizer (BertTokenizer): Pretrained BERT tokenizer.
            model (BertModel): Pretrained BERT model.
        """
        # Check for GPU availability and set the device accordingly.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the BERT tokenizer and model for 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def batch_embeddings(self, sentences):
        """
        Generates embeddings for a batch of sentences using the BERT model.
        
        Parameters:
            sentences (list): A list of sentences to embed.
        
        Returns:
            np.ndarray: A NumPy array of embeddings.
        """
        # Tokenize the input sentences and prepare them for the model.
        inputs = self.tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        # Get the model outputs (hidden states).
        outputs = self.model(**inputs)

        # Take the average of the last hidden states to get sentence embeddings.
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).detach().cpu().numpy()
        return embeddings
    
    def calculate_cosine_similarity(self, list1, list2, batch_size=32):
        """
        Calculates cosine similarities between two lists of sentences.
        
        Parameters:
            list1 (list): The first list of sentences.
            list2 (list): The second list of sentences, must be the same length as list1.
            batch_size (int): The size of batches to process. Defaults to 32.
        
        Returns:
            list: A list of cosine similarity scores for corresponding sentences.
        """
        # Initialize an empty list to store all similarity scores.
        total_similarities = []

        # Process the sentences in batches.
        for i in range(0, len(list1), batch_size):
            # Get the batches for both lists.
            batch1 = list1[i:i + batch_size]
            batch2 = list2[i:i + batch_size]

            # Compute embeddings for each batch.
            embeddings1 = self.batch_embeddings(batch1)
            embeddings2 = self.batch_embeddings(batch2)

            # Calculate cosine similarity for the embeddings of the current batch.
            similarities = cosine_similarity(embeddings1, embeddings2).diagonal()
            # Extend the total similarities list with the results from the current batch.
            total_similarities.extend(similarities)

        # Return the list of all similarity scores.
        return total_similarities
