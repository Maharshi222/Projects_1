import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("Ram")) == "Ram"




