import re

def encode(text, vocab):
    toks = re.split('([,.:;?_!"()\']|--|\s)', text)
    toks = [tok.strip() for tok in toks if tok.strip()]
    ids = [vocab[tok] for tok in toks]

    return ids


def decode(ids, vocab):
    inv_vocab = {v: k for k,v in vocab.items()}
    decoded_str = ""
    for id_val in ids:
        decoded_str+= inv_vocab[id_val]
        decoded_str+= " "
    decoded_str = re.sub(r'\s+([,.?!"()\'])', r'\1', decoded_str).strip()
    
    return decoded_str