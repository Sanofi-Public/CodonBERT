import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print(f"Loss after epoch {self.epoch}: {loss_now}")
        self.epoch += 1


class rna2vec:
    """Min class coontaining model training workflow for rnalm."""

    def __init__(self):
        """Config for gensim w2v"""
        self.param = {
            "sg": 1,
            "vector_size": 100,
            "window": 5,
            "min_count": 5,
            "negative": 2,
            "hs": 1,
            "seed": 56,
            "workers": 16,
            "alpha": 0.00025,
            "min_alpha": 1e-4,
        }
        self.epoch = 50

    def train_model(self, sentence_tokens):
        """
        Trains a Word2Vec model using the provided list of sentence tokens.
        Args:
            sentence_tokens (list): List of sentence tokens for vectorization.
        Returns:
            gensim.models.Word2Vec: The trained Word2Vec model.
        """
        model = gensim.models.Word2Vec(
            sentence_tokens, **self.param, callbacks=[callback()], compute_loss=True
        )
        model.train(sentence_tokens, total_examples=len(sentence_tokens), epochs=5)
        self.model = model
        return model

    @staticmethod
    def save(model, path="word2vec.model"):
        """save model file
        Args:
            model: artifact to save
            path: path to save model file
        """
        model.save(path)
