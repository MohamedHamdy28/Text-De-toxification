import pandas as pd

class DeletionModel:
    def __init__(self, bad_words_path):
        self.bad_words = pd.read_csv(bad_words_path, header=None)[0].tolist()

    def train(self):
        print("No need to train this model")

    def detoxify(self, X_test):
        result = []
        for sentence in X_test:
            result_sentence = []
            for word in sentence.split():
                if word not in self.bad_words:
                    result_sentence.append(word)
            result.append(" ".join(result_sentence))
        return result
