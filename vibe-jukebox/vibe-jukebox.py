###############################################################################
# Script for generating samples of music from a predicted artist/genre and
# generated lyrics using OpenAI's jukebox
#
# Adapted from:
# github.com/openai/jukebox/blob/08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3/jukebox/Interacting_with_Jukebox.ipynb
###############################################################################

import jukebox
import torch as t
import librosa
import os
from functools import reduce
from time import time
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
from joblib import load
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

###############################################################################
# Setup
###############################################################################

# Get distributed or cuda details
rank, local_rank, device = setup_dist_from_mpi()

# Set hyperparameters
with open("config.yml", 'r') as f:
    config_dict = yaml.load(f, Loader)
model = config_dict.get("model", "1b_lyrics")
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model=='5b_lyrics' else config_dict.get("num_samples", 8)
hps.name = 'samples'
chunk_size = 16 if model=="5b_lyrics" else 32
max_batch_size = 3 if model=="5b_lyrics" else 16
hps.levels = config_dict.get("upsample_levels", 3)
hps.hop_fraction = [.5,.5,.125]
multi_prompt = config_dict.get("multi_prompt", False)

# Set up vector quantizer
vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

# How long of a song to generate...OpenAI recommends 60 to 240 seconds, but of
# course generation time will be proportionally longer
sample_length_in_seconds = config_dict.get("song_length_sec", 60)
hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

###############################################################################
# Load prompt
###############################################################################

# Load the metadata prompt from the rest of the notebook
# This should be a dictionary with the following (key, val) pairs:
# {
#     "artist": PREDICTED ARTIST FROM NOTEBOOK
#     "genre": LIST OF 3 PREDICTED GENRES FROM NOTEBOOK
#     "lyrics": GENERATED LYRICS FROM GPT OR ELSEWHERE
# }
with open("prompt.yml", 'r') as f:
    metas_skel = yaml.load(f, Loader)

# Check skeleton has what we expect
keys = metas_skel.keys()
assert type(metas_skel) == type(dict()), "Loaded prompt object not a dictionary"
assert all(["artist" in keys, "genres" in keys, "lyrics" in keys]), "Loaded dictionary doesn't have the right keys"
assert type(metas_skel["genre"]) == type([]), "Genres must be in a list"
assert len(metas_skel["genre"]) > 0 and len(metas_skel["genre"]) <= 5, "Must specify 1 to 5 genres in genre list"

# Depending on if we're doing 1b or 5b param model, either keep first genre
# or combine the 3, respectively
if model=='5b_lyrics':
    metas_skel["genre"] = reduce(lambda x, y: x + " " + y, metas_skel["genre"])
else:
    metas_skel["genre"] = metas_skel["genre"][0]


# Add some extra details to the prompt, and make `n_samples` copies in a list
metas_skel["total_length"] = hps.sample_length
metas_skel["offset"] = 0
metas = [metas_skel] * hps.n_samples

# Generate batch of labels
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

###############################################################################
# First layer of music generation
###############################################################################

# OpenAI recoommends temperature of 0.98 or 0.99
sampling_temperature = config_dict.get("temperature", 0.985)

lower_batch_size = 16
max_batch_size = 3 if model == "5b_lyrics" else 16
lower_level_chunk_size = 32
chunk_size = 16 if model == "5b_lyrics" else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

# Generate the "base layer" of music, which will have general structure but
# noisy and mising detail until later upsampled
# This will take about 10 minutes per 30 seconds of requested length
print(80*'*')
print("Generating Layer 2...")
start = time()
zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
end = time()
print(f"Layer 2 complete! Took {start - end} seconds")

###############################################################################
# Upsample Stage for second and final layers
###############################################################################

# Set this False if you are on a local machine that has enough memory (this allows you to do the
# lyrics alignment visualization during the upsampling stage). For a hosted runtime, 
# we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
if True:
    del top_prior
    empty_cache()
    top_prior=None

# Load upsamplers
print("Loading upsamplers...")
upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]
print("Upsamplers loaded!")

# Upsample the base layer into better sounding music
print(80*'*')
print("Generating Layers 1 and 0...")
start = time()
zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)
end = time()
print(f"Layers 1 and 0 complete! Took {start - end} seconds")