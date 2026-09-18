import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.linear_op = F.linear

        synapse_mask = torch.zeros(
            (self.output_size * self.spines_per_output_channel, self.input_size)
        )
        for i in range(self.output_size * self.spines_per_output_channel):
            random_pos = torch.randint(low=0, high=self.input_size, size=([]))
            synapse_mask[i, random_pos] = 1
        self.register_buffer("synapse_mask", synapse_mask)

        self.weights = nn.Parameter(
            torch.ones(
                self.output_size * self.spines_per_output_channel, self.input_size
            ),
            requires_grad=False,
        )

    def rewire_synapses(self, rewiring_mask):
        repeated_rewiring_mask = rewiring_mask.repeat_interleave(
            self.spines_per_output_channel
        )
        for i in range(self.output_size * self.spines_per_output_channel):
            if repeated_rewiring_mask[i] == 1:
                # deleting old synapses
                self.synapse_mask[i, :] = 0

                # drawing new random synapses
                random_pos = torch.randint(low=0, high=self.input_size, size=([]))
                self.synapse_mask[i, random_pos] = 1

    def __call__(self, _input):
        masked_weights = self.weights * self.synapse_mask
        output = self.linear_op(_input, masked_weights)
        if self.reshape_output:
            output = torch.reshape(
                output,
                (*_input.shape[:-1], self.output_size, self.spines_per_output_channel),
            )

        return output
