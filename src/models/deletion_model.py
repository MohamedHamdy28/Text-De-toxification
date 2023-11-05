# Import the pandas library to handle data manipulation and CSV file reading.
import pandas as pd

class DeletionModel:
    """
    A simple model that deletes bad words from sentences.
    
    This model works by maintaining a list of words considered 'bad' or inappropriate.
    It processes text by removing these words. No machine learning training is involved since
    the operation is based solely on word deletion.
    
    Attributes:
        bad_words (list of str): A list of words that are to be filtered out of texts.
    """
    
    def __init__(self, bad_words_path):
        """
        The constructor for DeletionModel class.
        
        Parameters:
            bad_words_path (str): The file path to a CSV file containing a list of bad words.
        """
        # Read the bad words from a CSV file into a pandas DataFrame, then convert them into a list.
        self.bad_words = pd.read_csv(bad_words_path, header=None)[0].tolist()

    def train(self):
        """
        The training method for the DeletionModel.
        
        In this context, training is not required because the model's behavior is predefined
        by the bad words list and involves no statistical or machine learning methods.
        """
        # Since there is no training involved, we output a message stating that.
        print("No need to train this model")

    def detoxify(self, X_test):
        """
        Detoxify a list of sentences by removing bad words.
        
        Parameters:
            X_test (list of str): A list of sentences to be detoxified.
        
        Returns:
            result (list of str): The list of detoxified sentences.
        """
        # Initialize an empty list to store the results.
        result = []
        # Loop through each sentence in the input list.
        for sentence in X_test:
            # Initialize an empty list to store words that are not bad.
            result_sentence = []
            # Split the sentence into individual words and check each one.
            for word in sentence.split():
                # If the word is not in the list of bad words, add it to the results.
                if word not in self.bad_words:
                    result_sentence.append(word)
            # Join the words back into a sentence and add it to the list of results.
            result.append(" ".join(result_sentence))
        # Return the list of detoxified sentences.
        return result
