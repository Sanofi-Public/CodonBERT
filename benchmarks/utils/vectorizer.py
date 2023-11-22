import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


class Vectorizer(object):
    """
    Attributes:
        pipeline (list): A list of text objects for input.
        model (list): A hierarchy object derived from gensim.
        text (str): A distance matrix representing similarity.
        deviation_tokens (list): Tokens for the processing step at the beginning.
        stop_words (list): Stop words for the processing step at the beginning.
        rna_fragments (list): List of RNA fragments.

    Methods:
        get_average_vector_field(words): Calculates the average vector field for a given list of words.

    Usage:
        vector_space_embedder = Vectorizer()
        vector_space_embedder.model = mod
        vector_space_embedder.pipeline = nlp
        test_vector = vector_space_embedder.create_vector_from_model(values[0])
        vector_space_embedder.init_nlp_pipeline(values)
        vector_space_embedder.deviation_tokens[0]
    """

    def __init__(self):
        self.pipeline = []  #
        self.model = []  #
        self.text = ""
        self.rna_fragments = []

    def get_average_vector_field(self, words):
        size = len(words)
        start = np.zeros(self.model.vector_size)
        failures = 0
        fail = 0
        for word in words:
            try:
                start += self.model.wv.get_vector(word)
            except:
                failures += 1
        if failures == size:
            fail = 1

        # Create output for monitoring purpose
        output_dictionary = {
            "vector": start / size,
            "completness": failures,
            "fail": fail,
            "empty": 0,
        }
        dummy = np.zeros(self.model.vector_size)

        if np.isnan(size):
            output_dictionary["vector"] = dummy
            return output_dictionary

        if np.isnan(start).any():
            output_dictionary["vector"] = dummy
            return output_dictionary

        if size == 0:
            output_dictionary["empty"] = 1
            output_dictionary["vector"] = dummy
            return output_dictionary

        if np.isnan(start / size).any():
            output_dictionary["vector"] = dummy

        return output_dictionary

    def create_tfidf_stack(self, _texts, **args):
        """
        Create a TF-IDF embedding for a given list of tokenized texts.
        Args:
            _texts (list): A list of token lists.
            **args: Additional arguments to be passed to the TfidfVectorizer.
        Returns:
            tuple: A tuple containing the vectorizer, TF-IDF matrix, and the matrix in array format.
        """
        _texts = [" ".join(kmer_list) for kmer_list in _texts]
        vectorizer = TfidfVectorizer(**args)
        docs_tfidf = vectorizer.fit_transform(_texts)
        return vectorizer, docs_tfidf, docs_tfidf.toarray()

    def create_vector_stack(self):
        """
        Creates a vector array that is easy to use with scikit-learn.
        Returns:
            numpy.ndarray: The mean vector array.
        """
        holo_vector_list = [
            self.get_average_vector_field(entry).get("vector")
            for entry in self.rna_fragments
        ]
        vector_matrix = np.stack(holo_vector_list)
        return vector_matrix.astype("float32")

    def create_vector_concat(self, pad=True):
        """
        Create a vector array and concatenate all word vectors for each fragment. Optionally pad arrays at the end.
        Args:
            pad (bool, optional): Whether to pad arrays at the end. Defaults to True.
        Returns:
            numpy.ndarray: The concatenated vector array.
        """
        out = [
            np.array([self.model.wv.get_vector(word) for word in entry])
            for entry in self.rna_fragments
        ]
        if pad:
            n_rows = max([x.shape[0] for x in out])

            def pad_row(x):
                return np.pad(x, pad_width=((0, n_rows - x.shape[0]), (0, 0)))

            out = [pad_row(x) for x in out]
            out = np.array(out)
        else:
            out = np.array(out, dtype=object)
        return out
