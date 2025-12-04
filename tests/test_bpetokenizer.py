from llmmini.bpetokenizer import BPETokenizer

tok = BPETokenizer("gpt2")
print("vocab_size:", tok.vocab_size)

text = "Hello world!"
ids = tok.encode(text)
print(ids)

print(tok.decode(ids))
