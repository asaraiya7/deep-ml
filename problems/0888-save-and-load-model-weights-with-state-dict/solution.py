import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    file_buffer = io.BytesIO()
    torch.save(src.state_dict(), file_buffer)

    # Point buffer to start of the file, points to end of dict by default
    file_buffer.seek(0)
    state_dict = torch.load(file_buffer)
    dst.load_state_dict(state_dict)

    return dst