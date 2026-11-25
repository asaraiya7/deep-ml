def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    dec_str = ""
    for id_x in ids: 
        if vocab[id_x].startswith("G"):
            # Add a space and then skip first letter (reference approach to skip char)
            dec_str+= " " + vocab[id_x][1:]
        else:
            dec_str+= vocab[id_x]
    return dec_str
