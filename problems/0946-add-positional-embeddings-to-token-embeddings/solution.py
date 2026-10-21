import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    # Token embeddings: [B, seq_len, embed_dim]
    # Pos embeddings: [context_len, embed_dim]
    # context_len > seq_len
    seq_len = token_embeddings.shape[1]
    seq_len_pos_embed_matrix = pos_embedding_matrix[:seq_len, :]
    # Add along axis = 1 (checks shape from RHS and automatically broadcasts B dimension)
    token_embeddings+= seq_len_pos_embed_matrix
    return token_embeddings