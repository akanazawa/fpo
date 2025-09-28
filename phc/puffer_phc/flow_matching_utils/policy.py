import torch

class FlowMatchingPolicy(torch.nn.Module):
    '''Wrap a non-recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
         action, value = self.policy(x)
         logprob = torch.zeros_like(action[:, 0])
         entropy = torch.zeros_like(action[:, 0])
         return action, logprob, entropy, value

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)
