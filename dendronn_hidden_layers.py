import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")


class DendroNNHiddenLayer(nn.Module):
    def __init__(
        self, input_size, output_size, spines_per_output_channel, reshape_output=True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.spines_per_output_channel = spines_per_output_channel
        self.reshape_output = reshape_output

        num_output_spines = self.output_size * self.spines_per_output_channel
        synapse_indices = torch.randint(
            low=0, high=self.input_size, size=(num_output_spines,)
        )
        self.register_buffer("synapse_indices", synapse_indices)

    def rewire_synapses(self, rewiring_mask):
        repeated_rewiring_mask = rewiring_mask.repeat_interleave(
            self.spines_per_output_channel
        )
        rewired_rows = repeated_rewiring_mask.nonzero(as_tuple=True)[0]
        if rewired_rows.numel() > 0:
            new_indices = torch.randint(
                low=0,
                high=self.input_size,
                size=(rewired_rows.numel(),),
                device=self.synapse_indices.device,
            )
            self.synapse_indices[rewired_rows] = new_indices

    def forward(self, _input):
        output = _input.index_select(-1, self.synapse_indices)
        if self.reshape_output:
            output = torch.reshape(
                output,
                (*_input.shape[:-1], self.output_size, self.spines_per_output_channel),
            )

        return output
