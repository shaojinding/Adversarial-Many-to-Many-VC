from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
import numpy as np
import librosa
import argparse
import shutil
from pathlib import Path
from utils.argutils import print_args
import random
import soundfile
encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/checkpoint_200k_steps.pt")
syn_dir = Path("synthesizer/saved_models/logs-train_adversarial_vctk/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

def synthesize_ppg(ppg, embed):
    specs = synthesizer.synthesize_spectrograms([ppg], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav

def synthesize_ppg_batch_random_embed(ppg_paths, embed_paths):
    embed_idx = random.randint(0, len(embed_paths) - 1)
    embed = np.load(embed_paths[embed_idx])

    generated_wavs = []

    for ppg_path in ppg_paths:
        ppg = np.load(ppg_path)
        generated_wavs.append(synthesize_ppg(ppg, embed))

    return generated_wavs, embed_idx

def synthesize_ppg_batch_avg_embed(ppg_paths, embed_paths):
    embeds = np.zeros((len(embed_paths), 256), dtype=np.float32)
    for i, p in enumerate(embed_paths):
        embeds[i, :] = np.load(p)
    avg_embed = embeds.mean(axis=0)

    generated_wavs = []

    for ppg_path in ppg_paths:
        ppg = np.load(ppg_path)
        generated_wavs.append(synthesize_ppg(ppg, avg_embed))

    return generated_wavs

def parse_meta(synthesizer_root, meta_file_path):
    with open(meta_file_path, 'r') as f:
        lines = f.readlines()
    wav_path_dict = {}
    ppg_path_dict = {}
    embed_path_dict = {}
    for line in lines:
        meta = line.strip().split('|')
        wav_fname = meta[0]
        ppg_fname = meta[2]
        embed_fname = meta[3]
        speaker = ppg_fname.split('-')[1]

        wav_path_dict[synthesizer_root.joinpath('audio', wav_fname)] = speaker
        ppg_path_dict[synthesizer_root.joinpath('ppgs', ppg_fname)] = speaker
        embed_path_dict[synthesizer_root.joinpath('embeds', embed_fname)] = speaker

    speakers = sorted(list(dict.fromkeys(ppg_path_dict.values())))
    return wav_path_dict, ppg_path_dict, embed_path_dict, speakers




def save_wavs(wav_paths, target_wav_paths, generated_wavs_avg, generated_wavs_random, embed_idx, output_dir):
    assert len(wav_paths) == len(generated_wavs_avg) == len(generated_wavs_random)
    for i, wav_path in enumerate(wav_paths):
        basename = wav_path.stem
        wav = np.load(wav_path)
        target_wav = np.load(target_wav_paths[embed_idx])
        soundfile.write(output_dir.joinpath(basename + '_src.wav'), wav,
                        synthesizer.sample_rate)
        soundfile.write(output_dir.joinpath(basename + '_tgt.wav'), target_wav,
                        synthesizer.sample_rate)
        soundfile.write(output_dir.joinpath(basename + '_vc_avg.wav'), generated_wavs_avg[i],
                                 synthesizer.sample_rate)
        soundfile.write(output_dir.joinpath(basename + '_vc_rdm.wav'), generated_wavs_random[i],
                                 synthesizer.sample_rate)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("synthesizer_root", type=Path, help= \
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("output_dir", type=Path, help= \
        "Path to the output_synthesis")
    parser.add_argument("--num_speakers", type=int, default=4, help= \
        "Number of speakers used for testing")
    parser.add_argument("--num_utterances", type=int, default=4, help= \
        "Number of utterances per speaker used for testing")

    args = parser.parse_args()
    print_args(args, parser)

    meta_file_path = args.synthesizer_root.joinpath('train.txt')

    wav_path_dict, ppg_path_dict, embed_path_dict, speakers = parse_meta(args.synthesizer_root, meta_file_path)


    speakers = speakers[:args.num_speakers]
    for speaker in speakers:
        wav_paths = sorted([k for k, v in wav_path_dict.items() if v == speaker])[:args.num_utterances]
        ppg_paths = sorted([k for k, v in ppg_path_dict.items() if v == speaker])[:args.num_utterances]
        for target_speaker in speakers:
            pair_dir = args.output_dir.joinpath('{}-{}'.format(speaker, target_speaker))
            print(pair_dir)
            pair_dir.mkdir(parents=True, exist_ok=True)
            embed_paths = sorted([k for k, v in embed_path_dict.items() if v == target_speaker])
            target_wav_paths = sorted([k for k, v in wav_path_dict.items() if v == target_speaker])
            generated_wavs_avg = synthesize_ppg_batch_avg_embed(ppg_paths, embed_paths)
            generated_wavs_random, embed_idx = synthesize_ppg_batch_random_embed(ppg_paths, embed_paths)
            save_wavs(wav_paths, target_wav_paths, generated_wavs_avg, generated_wavs_random, embed_idx, pair_dir)


if __name__ == '__main__':
    main()
