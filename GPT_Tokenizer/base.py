"Few Helper function for tokenizer"

import unicodedata

def get_stats(ids,counts=None):
    "Takes list of integers and returns a dictionary of count of consecutive pair"
    "Eg-list1 = [1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}"

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts

def merge(ids,pair,idx):
    """
    As per BPE, replacing all the repeating pairs from ids, with idx
    Eg-replacing with idx=4, above list becomes = [4,3,4]
    ids = [1,2,3,1,2], pair = (1,2), idx = 4 thus output = [4,3,4]
    """

    newids = []
    i=0
    while i < len(ids):
        #condition is, if its not at the very last position & their pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
        #checking if first position from both pair and ids matches or not,
        #then checks the lenght and then checks the second position of pair that matches or not
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids

def replace_control_characters(s:str) -> str:
    """
    Not to print control characters which will distort output
    control characters are: /n, /l, etc
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
            #this character is okay
        else:
            chars.append(f"\\u{ord(ch):04x}") #escape this
    return "".join(chars)


def render_token(t:bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

#-----------------------------------------------------------------------------------------------------------------
#The Base Tokenier Class

class Tokenizer:
    """ The Base class for Tokenizer """

    def __init__(self):
        #keeping the vocab size of 256, no merge, no pattern
        self.merges = {} #(int,int) -> int
        self.pattern = "" #str
        self.special_tokens = {} # str->int i.e. ("<|endoftext|>" : 100257)
        self.vocab = self._build_vocab() #int -> bytes

    def train(self, text, vocab_size, verbose=False):
        #Tokenizer can train a vocabulary of size vocab_size from given text
        raise NotImplementedError
    def encode(self, text):
        #Tokenizer can encode a string into a list of integers
        raise NotImplementedError
    def decode(self, ids):
        #Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    def _build_vocab(self):
        #vocab os simple and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    def save(self,file_prefix):
        """
        this saves two files: file_prefix.vocab and file_prefix.model
        model file is for load and vocab file is for just inspection
        """

        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("GPT_Tokenizer v1\n")
            f.write(f"{self.pattern}\n")
            #write the special tokens endtext annd all, first the number then each token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special}{idx}\n")
            #merging the dictionary
            for idx1 , idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
            #write the vocab, for humans to look at
            vocab_file = file_prefix + ".vocab"
            inverted_merges = {idx : pair for pair, idx in self.merges.items()}
            with open(vocab_file, "w", encoding="utf-8") as f:
                for idx, token in self.vocab.items():
                    """note: many tokens may be partial utf encoded
                    so cannot be decoded into valid string.
                    """
                    s = render_token(token)

                    if idx in inverted_merges:
                        idx0, idx1 = inverted_merges[idx]
                        s0 = render_token(self.vocab[idx0])
                        s1 = render_token(self.vocab[idx1])
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        f.write(f"[{s}] {idx}\n")
    def load(self, model_file):
        assert model_file.endswith(".model")
        #reads the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            #read the version
            version = f.readline().strip()
            assert version == "GPT_Tokenizer v1"
            #read the pattern
            self.pattern = f.readline().strip()
            #read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

                    
            
