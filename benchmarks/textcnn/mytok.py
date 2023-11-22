from collections import Counter
from itertools import islice


class KmerFragmenter:
    """Slide only a single nucleotide"""

    def __init__(self):
        self.k_low = 3
        self.k_high = 8

    def sliding_random(self, rng, sequence):
        """
        Generates sliding windows of length k from a given sequence with a random starting position.
        Args:
            seq (str): The input sequence.
            k (int): The length of the sliding window.
            step (int): The step size for sliding the window.
        Returns:
            List[str]: A list of sliding windows generated from the sequence.
        Raises:
            ValueError: If the length of the sequence is less than the window size.
        """
        sentence_tokens = list()
        for seq in sequence:
            sentence_tokens.append(
                list(
                    seq[i : i + rng.randint(self.k_low, self.k_high + 1)]
                    for i in range(len(seq) - self.k_high + 1)
                )
            )
        return sentence_tokens

    @staticmethod
    def random_chunks(rng, li, min_chunk, max_chunk):
        """
        Generates random chunks of elements from a given list within a specified length range.
        Args:
            rng (numpy.random.Generator): Random number generator object.
            li (Iterable): The input list or iterable.
            min_chunk (int): The minimum length of the generated chunks.
            max_chunk (int): The maximum length of the generated chunks.
        Yields:
            str: Random chunks of elements from the input list."""
        it = iter(li)
        while True:
            head_it = islice(it, rng.randint(min_chunk, max_chunk + 1))
            nxt = "".join(head_it)
            # throw out chunks that are not within the kmer range
            if len(nxt) >= min_chunk:
                yield nxt
            else:
                break

    def disjoint_random(self, rng, sequence):
        """
        Generates disjoint random chunks of length k from a given sequence.
        Args:
            seq (str): The input sequence.
            k (int): The length of each chunk.
            num_chunks (int): The number of chunks to generate.
        Returns:
            List[str]: A list of disjoint random chunks generated from the sequence.
        Raises:
            ValueError: If the length of the sequence is less than the total length of all chunks.

        """
        sentence_tokens = list()
        for seq in sequence:
            seq = seq[rng.randint(self.k_low) :]
            sentence_tokens.append(
                list(self.random_chunks(rng, seq, self.k_low, self.k_high))
            )
        return sentence_tokens

    def split_words(self, sequences, kmer_len, s):
        """
        Splits words in equal chunks.

        Args:
            sequences (List[str]): List of sequences.
            kmer_len (int): Fixed size of the kmers.
            s (int): Increments.

        Returns:
            List[str]: List of strings.
        """
        out = []
        for i in sequences:
            kmer_list = []
            for j in range(0, (len(i) - kmer_len) + 1, s):
                kmer_list.append(i[j : j + kmer_len])
            out.append(kmer_list)
        return out
