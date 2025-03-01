from typing import Sequence
import torch


def get_max_spec_size(config):
    # return (T, F)
    return (
        (
            config.datamodule.window_length
            + sum(config.datamodule.padding)
            - config.logspec.n_fft
        )
        // config.logspec.hop_length
        + 1,
        config.logspec.n_fft // 2 + 1,
    )


def get_custom_collate(max_T: int = 622, is_test: bool = False):
    """Creates a custom collate function for the dataloader.

    This function pads the input sequences to the same length `max_T`
    and returns a dictionary containing the padded input sequences,
    target sequences, and their corresponding lengths.

    Args:
        max_T (int): The maximum length to pad the input sequences to.

    Returns:
        callable: A custom collate function that can be used with
            the dataloader.
    """

    def custom_collate(
        samples: Sequence[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        # inputs: (T, num_bands, num_channels, F)
        inputs = [sample[0].movedim(0, -1) for sample in samples]  # [(num_bands, num_channels, F, T) x N]
        targets = [sample[1] for sample in samples]  # [(T,)]

        # Batch of inputs and targets padded along time
        if not is_test:
            input_batch = torch.zeros(len(inputs), *inputs[0].shape[:-1], max_T)
            for i, input in enumerate(inputs):
                input_batch[i, :, :, :, :] = torch.nn.functional.pad(
                    input, (0, max_T - input.shape[-1]), mode="constant", value=0
                )
        else:
            input_batch = inputs[0].unsqueeze(0)

        input_batch = input_batch.movedim(-1, 0)  # (T, N, ..., F)
        target_batch = torch.nn.utils.rnn.pad_sequence(targets)  # (T, N)

        # Lengths of unpadded input and target sequences for each batch entry
        input_lengths = torch.as_tensor(
            [_input.shape[-1] for _input in inputs], dtype=torch.int32
        )
        target_lengths = torch.as_tensor(
            [len(target) for target in targets], dtype=torch.int32
        )

        return {
            "inputs": input_batch,
            "targets": target_batch,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
        }

    return custom_collate
