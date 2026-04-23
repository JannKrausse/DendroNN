import torch
import torch.nn as nn

from dendronn_unit import DendroNNUnit
from integrator_unit import SimpleQuantizedIntegrator
from dendronn_hidden_layers import DendroNNHiddenLayer


class DendroNN(nn.Module):
    def __init__(
            self,
            num_spines,
            num_hidden_layer,
            num_hidden_units,
            max_seq_len,
            in_shape,
            out_shape,
            batch_size,
            hyperparameters,
            spike_acceptance_window=2,  # max_time to receive input spike
            dropout_p=0.,
            bias=False,
            min_num_spines=None,  # lower bound for possible sequence lengths in case they are distributed
            refrac_period=False,
            parallel_sequences=True,
    ):
        super().__init__()
        self.num_spines = num_spines
        self.min_num_spines = min_num_spines
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_units = num_hidden_units
        self.max_seq_len = max_seq_len
        self.spike_acceptance_window = spike_acceptance_window
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.batch_size = batch_size
        self.hyperparameters = hyperparameters
        self.refrac_period = refrac_period
        self.parallel_sequences = parallel_sequences

        self.max_seq_len_out = max_seq_len

        # initialization of inter-spike intervals
        delay_size = (num_hidden_layer, num_hidden_units, max(num_spines - 1, 1))
        effective_seq_len = self.spike_acceptance_window * num_spines  # (max_)seq_len without inhibition / delay
        self.inter_spike_intervals = torch.randint(low=0,
                                                high=max(int((hyperparameters["max_seq_len"] -
                                                        effective_seq_len) / (max(num_spines - 1, 1))), 1),
                                                size=delay_size)
        self.max_seq_len = self.inter_spike_intervals.sum(-1) + effective_seq_len

        self.hidden_layer = nn.ModuleDict()
        self.units = nn.ModuleDict()
        for i in range(self.num_hidden_layer):
            if i == 0:
                num_in_units = in_shape
            else:
                num_in_units = self.num_hidden_units
            self.hidden_layer[str(i)] = DendroNNHiddenLayer(num_in_units,
                                                             self.num_hidden_units,
                                                             self.num_spines)
            inter_spike_intervals_ = self.inter_spike_intervals[i, :]
            max_seq_len_ = self.max_seq_len[i, :]
            self.units[str(i)] = DendroNNUnit(
                num_spines=num_spines,
                num_units=num_hidden_units,
                batch_size=batch_size,
                max_seq_len=max_seq_len_,
                inter_spike_intervals=inter_spike_intervals_,
                spike_acceptance_window=spike_acceptance_window,
                min_num_spines=min_num_spines,
                parallel_seqs=parallel_sequences,
            )

        if self.num_hidden_layer != 0:
            num_in_units = self.num_hidden_units
        else:
            num_in_units = in_shape
        self.hidden_layer["out"] = nn.Linear(num_in_units, out_shape, bias=bias)
        self.units["out"] = SimpleQuantizedIntegrator(quantize=False,  # quantization will be enabled later during compression
                                                      bit_width=16,
                                                      symmetric=True)

        self.testing = False
        self.dropout = nn.Dropout(dropout_p)

        self.device = None

    def forward(self, x):
        seq_len, batch_size, in_shape = x.shape
        self.device = x.device

        z, spikes_hidden = self.hidden_pass(x, batch_size, seq_len)

        predictions, spikes_output = self.output_pass(z, seq_len)

        if not spikes_output == []:
            spikes = torch.stack([spikes_hidden, spikes_output.unsqueeze(dim=1)])  # dim=1 is layer dim of spikes_hidden
        elif not spikes_hidden == []:
            spikes = spikes_hidden
        else:
            spikes = torch.zeros((seq_len, self.num_hidden_layer, batch_size, self.num_hidden_units))

        return predictions, spikes.to(predictions.device)

    def hidden_pass(self, x, batch_size, seq_len):
        x = x.to(torch.float32)

        if x.max() > 1.0:
            raise ValueError("Input values should be binary.")

        refrac_period_mask = torch.ones((batch_size, self.num_hidden_units), device=x.device)

        output = []
        spikes = []
        for u in self.units.values():  # in case of irregular batch_size
            if isinstance(u, DendroNNUnit):
                u.batch_size = batch_size

        self.reset_units()  # reset all tensors and buffers for next inference

        for i in range(seq_len):
            z = x[i, :, :]
            spikes_tmp = []
            for j in range(self.num_hidden_layer):
                z = self.hidden_layer[str(j)](z)
                z = self.units[str(j)](z)
                if self.refrac_period:
                    z = z * refrac_period_mask
                if z.dim() == 0:
                    z = z.unsqueeze(dim=0)

                spikes_tmp.append((z != 0).to(torch.float))
                refrac_period_mask = refrac_period_mask * (1 - z.detach())
            output.append(z)
            if not spikes_tmp == []:
                spikes.append(torch.stack(spikes_tmp))
        if not spikes == []:
            spikes = torch.stack(spikes)
        output = torch.stack(output)

        return output, spikes

    def output_pass(self, x, seq_len):
        predictions = []
        spikes = []
        mem = self.units["out"].reset_mem()

        for i in range(seq_len):
            z = x[i, :, :]
            z = self.hidden_layer["out"](z)

            z = self.dropout(z)
                
            _, mem = self.units["out"](z, mem)  # first output is spikes of parent class of integrator unit
            z = mem

            # z = torch.nn.functional.softmax(z, dim=-1)

            predictions.append(z)
        predictions = torch.stack(predictions)
        if not spikes == []:
            spikes = torch.stack(spikes)
        return predictions, spikes

    def reset_units(self):
        for d in self.units.values():
            if isinstance(d, DendroNNUnit):
                d.reset(self.device)
