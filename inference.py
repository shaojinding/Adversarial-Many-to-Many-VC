from IPython.display import Audio
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-train_ppg/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

ppg = np.load('/data/AutoSpeech/vc/LibriSpeech/SV2TTS/synthesizer/ppgs/ppg-61-70968-0000_00.npy')
embed = np.load('/data/AutoSpeech/vc/LibriSpeech/SV2TTS/synthesizer/embeds/ppg-908-31957-0015_00.npy')
specs = synthesizer.synthesize_spectrograms([ppg], [embed])