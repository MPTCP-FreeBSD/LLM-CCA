import torch.nn as nn
import torch

class EncoderNetwork(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for numerical features
        self.rtt_fc = nn.Linear(1, embed_dim)  # For RTT
        self.cwnd_fc = nn.Linear(1, embed_dim)  # For CWND
        self.throughput_fc = nn.Linear(1, embed_dim)  # For Throughput
        
        # Final fully connected layer to combine all features
        self.fc_final = nn.Linear(embed_dim * 3, embed_dim)  # 3 inputs: RTT, CWND, Throughput

    def forward(self, state):
        # Extract the components from the state
        rtt = state[..., 0].unsqueeze(-1)  # RTT
        cwnd = state[..., 1].unsqueeze(-1)  # CWND
        throughput = state[..., 2].unsqueeze(-1)  # Throughput

        # Process numerical features
        rtt_encoded = torch.relu(self.rtt_fc(rtt))
        cwnd_encoded = torch.relu(self.cwnd_fc(cwnd))
        throughput_encoded = torch.relu(self.throughput_fc(throughput))

        # Concatenate all encoded features
        combined = torch.cat([rtt_encoded, cwnd_encoded, throughput_encoded], dim=-1)

        # Final output
        output = torch.relu(self.fc_final(combined))

        return output

