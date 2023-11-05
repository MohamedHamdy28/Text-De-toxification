from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
import torch

class StyleTransferAccuracy:
    def __init__(self):
        """
        The initializer for the StyleTransferAccuracy class.
        
        Attributes:
            device (torch.device): Device configuration to use GPU if available.
            tokenizer (RobertaTokenizer): Pretrained RoBERTa tokenizer.
            model (RobertaForSequenceClassification): Pretrained RoBERTa model for sequence classification.
        """
        # Check if a GPU is available and set the device accordingly.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pretrained RoBERTa tokenizer and model for toxicity classification.
        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

        # Move the model to the configured device (GPU if available).
        self.model.to(self.device)

    def classify_preds(self, batch_size, preds):
        """
        Classifies predictions into styles using the RoBERTa model for sequence classification.
        
        Parameters:
            batch_size (int): The size of the batches to process.
            preds (list): A list of text predictions to classify.
        
        Returns:
            list: A list of classification results, where 1 represents 'non-toxic' and 0 represents 'toxic'.
        """
        print('Calculating style of predictions')
        results = []

        # Process the predictions in batches using tqdm for a progress bar.
        for i in tqdm.tqdm(range(0, len(preds), batch_size)):
            # Tokenize the batch of predictions and pad them for consistent length.
            batch = self.tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)

            # Move the tokenized batch to the device (GPU if available).
            batch = {key: value.to(self.device) for key, value in batch.items()}

            # Pass the batch through the model and get the logits.
            logits = self.model(**batch)['logits']
            # Determine the argmax of the logits to get the most likely class.
            # The argmax is converted to float and then to a list.
            result = logits.argmax(1).float().data.tolist()
            # The results are expected to be '1' for 'non-toxic' and '0' for 'toxic',
            # so we subtract from 1 since the RoBERTa model outputs '1' for 'toxic'.
            results.extend([1 - item for item in result])

        # Return the list of results with the style classifications.
        return results
