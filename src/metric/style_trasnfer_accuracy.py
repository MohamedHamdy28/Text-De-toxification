from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
import torch

class StyleTransferAccuracy:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        
        # Move the model to the GPU
        self.model = self.model.to(self.device)

    def classify_preds(self, batch_size, preds):
        print('Calculating style of predictions')
        results = []

        for i in tqdm.tqdm(range(0, len(preds), batch_size)):
            batch = self.tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)

            # Move the batch to the GPU
            batch = {key: value.to(self.device) for key, value in batch.items()}

            result = self.model(**batch)['logits'].argmax(1).float().data.tolist()
            results.extend([1 - item for item in result])

        return results
