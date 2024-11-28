# #Checking the replacement of control characters like new line/n  ot tab /t.
# # import unicodedata

# # def replace_control_characters(s: str) -> str:
# #     """
# #     Replace control characters in a string with their escaped Unicode representation.
# #     Control characters are like \n, \t, etc.
# #     """
# #     chars = []  # Fixing the typo here, initializing chars list correctly
# #     for ch in s:
# #         if unicodedata.category(ch)[0] != "C":  # Not a control character
# #             chars.append(ch)  # This character is okay
# #         else:
# #             chars.append(f"\\u{ord(ch):04x}")  # Escape this control character
# #     return "".join(chars)

# # # Test the function with a string that contains control characters
# # test_string = "Hello\nWorld\t!"
# # result = replace_control_characters(test_string)
# # print(result)
# ############################################################################################################
# #Checking code for Making A vocabulary using merge with special tokens
# class MyModel:
#     def __init__(self, merges, special_tokens):
#         self.merges = merges
#         self.special_tokens = special_tokens

#     def _build_vocab(self):
#         vocab = {idx: bytes([idx]) for idx in range(256)}

#         # Add merged tokens
#         for (p0, p1), idx in self.merges.items():
#             vocab[idx] = vocab[p0] + vocab[p1]

#         # Add special tokens
#         for special, idx in self.special_tokens.items():
#             vocab[idx] = special.encode("utf-8")

#         return vocab

# # Example usage
# merges = {
#     (0, 1): 2,
#     (0, 2): 3,
#     (1, 2): 4
# }

# special_tokens = {
#     "<unk>": 0,
#     "<s>": 1,
#     "</s>": 2
# }

# model = MyModel(merges, special_tokens)
# vocab = model._build_vocab()

# print(vocab)

######################################################################################################
########################################################################################################

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from minimal import BasicTokenizer
# # Sample text
# text = "abcabdacbabdca."
# # Initialize the tokenizer
# tokenizer = BasicTokenizer()
# # Set vocabulary size
# vocab_size = 300  # for example, 300 tokens
# # Train the tokenizer
# tokenizer.train(text, vocab_size, verbose=True)


#############################################################################################################
#############################################################################################################

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from minimal import BasicTokenizer

# txt = "RLS can be used to limit the data that users see based on their roles"
# tokenizer = BasicTokenizer()
# encoded_txt = tokenizer.encode(txt)
# print(encoded_txt)
# decoded_txt = tokenizer.decode([82,76])
# print(decoded_txt)

