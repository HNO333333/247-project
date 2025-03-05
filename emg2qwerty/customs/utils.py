import torch


def segment_stack(x: torch.Tensor, max_len: int, seg_axis: int = 0, stack_axis: int = 1):
    n_seg = (x.shape[seg_axis] + max_len - 1) // max_len
    pad_len = n_seg * max_len - x.shape[seg_axis]

    # Use torch.nn.functional.pad for padding
    padding_shape = list(x.shape)
    padding_shape[seg_axis] = pad_len
    padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)

    x_padded = torch.cat([x, padding], dim=seg_axis)

    # Use reshape and stack for segmentation
    slices = x_padded.split(max_len, dim=seg_axis)
    x_out = torch.stack(slices, dim=stack_axis)
    return x_out