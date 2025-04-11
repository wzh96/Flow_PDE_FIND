import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_encoder_layers, dim_feedforward, dropout, output_size):
        """
            Args:
                input_size (int): Number of input features per time step.
                d_model (int): The dimension of the embedding space (transformer's model dimension).
                num_heads (int): Number of heads in the multi-head attention.
                num_encoder_layers (int): Number of Transformer encoder layers.
                dim_feedforward (int): Dimension of the feedforward network inside the encoder.
                dropout (float): Dropout rate.
                output_size (int): Number of output features (e.g., predicting traffic flow for each sensor).
        """
        super(Transformer, self).__init__()
        # Project input to d_model dimension.
        self.input_linear = nn.Linear(input_size, d_model)

        # Define a single encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first = True)
        # Stack multiple encoder layers.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final fully connected layer to map d_model to the output size.
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x):
        """
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Project the input: (batch_size, sequence_length, d_model)
        x = self.input_linear(x)

        # PyTorch's Transformer expects input in shape (sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Pass through the Transformer encoder.
        x = self.transformer_encoder(x)

        # Use the output from the last time step for prediction.
        out = x[-1, :, :]  # Shape: (batch_size, d_model)
        out = self.fc(out)  # Shape: (batch_size, output_size)
        return out
