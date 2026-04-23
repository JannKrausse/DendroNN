import torch
import torch.nn as nn


class SimpleQuantizedIntegrator(nn.Module):
    def __init__(self, quantize=False, bit_width=8, symmetric=False, leaky=False, gamma=0.5866):
        super(SimpleQuantizedIntegrator, self).__init__()
        self.quantize = quantize
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.leaky = leaky
        self.gamma = gamma

        self._init_state()
        
        # Quantization scale and zero point (computed dynamically)
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("zero_point", torch.tensor(0.0), persistent=False)


    def _init_state(self):
        state = torch.zeros(0)
        self.register_buffer("state", state, False)

    def reset_mem(self):
        self.state = torch.zeros_like(self.state, device=self.state.device)
        return self.state
    
    def _quantize_state(self, state):
        if not self.quantize:
            return state
        
        n_levels = 2 ** self.bit_width
        
        if self.symmetric:
            # Symmetric quantization around zero
            abs_max = torch.max(torch.abs(state))
            if abs_max < 1e-8:
                return state
            
            self.scale = (2 * abs_max) / (n_levels - 1)
            quantized = torch.round(state / self.scale) * self.scale
            quantized = torch.clamp(quantized, -abs_max, abs_max)
        else:
            # Asymmetric quantization
            min_val = state.min()
            max_val = state.max()
            
            if torch.abs(max_val - min_val) < 1e-8:
                return state
            
            weight_range = max_val - min_val
            self.scale = weight_range / (n_levels - 1)
            self.zero_point = torch.round(-min_val / self.scale)
            
            # Quantize: scale, shift, round, then reverse
            quantized_int = torch.round(state / self.scale + self.zero_point)
            quantized_int = torch.clamp(quantized_int, 0, n_levels - 1)
            quantized = (quantized_int - self.zero_point) * self.scale
        
        return quantized
    
    def forward(self, input_, state):
        if not state == None:
            self.state = state

        if not self.state.shape == input_.shape:
            self.state = torch.zeros_like(input_, device=self.state.device)

        # Accumulate input into state and leak if necessary
        if not self.leaky:
            self.state = self.state + input_
        else:
            self.state = self.gamma * self.state + (1 - self.gamma) * input_
        
        # Apply quantization if enabled
        if self.quantize:
            self.state = self._quantize_state(self.state)
        
        return None, self.state
