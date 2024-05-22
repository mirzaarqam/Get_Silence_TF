import torch
import pickle

# Load the Silero VAD model and utilities from the repo
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')

# Save the model as a TorchScript model
torch.jit.save(model, 'silero_vad_model_jit.pt')

# Save the utilities using pickle
with open('silero_vad_utils.pkl', 'wb') as f:
    pickle.dump(utils, f)
