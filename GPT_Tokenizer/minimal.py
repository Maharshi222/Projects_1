"""
Minimal Byte Level Byte Pair Encoding Tokenizer
It does not handle split patterns and Special Tokens
"""

from base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size-256

        #input text preprocessing
        text_bytes = text.encode("utf-8") #raw bytes
        ids = list(text_bytes)#list of integers in range 0..255

        #iteratively merge the most common pair to create new tokens
        merges = {} #(int,int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} #int to -> bytes

        for i in range(num_merges): #count up the number every time a consecutive pair is seen/found
            stats = get_stats(ids) #
            pair = max(stats,key = stats.get)
            idx = 256+i
            ids = merge(ids,pair,idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurences")

        self.merges = merges #will be used in encode
        self.vocab = vocab#will be used in decode

    def decode(self,ids):
        #give here the ids( list of intergers) and return string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace") 
        return text
    
    def encode(self, text):
        #given string and returns token ids(ints)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
                        # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
