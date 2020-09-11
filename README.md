# Improving the Speaker Identity of Non-Parallel Many-to-Many Voice Conversion with Adversarial Speaker Recognition

Code for this paper [Improving the Speaker Identity of Non-Parallel Many-to-Many Voice Conversion with Adversarial Speaker Recognition](https://psi.engr.tamu.edu/wp-content/uploads/2020/08/IS2020_shaojin_Adversarial_speaker_classifier_camera_ready.pdf)

Shaojin Ding, Guanlong Zhao, Ricardo Gutierrez-Osuna

Accepted by INTERSPEECH 2020

This is a TensorFlow + Pytorch implementation. This implementation is adapted from the Real Time Voice Clone implementation at [https://github.com/CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

## Dataset:

* [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651)
  * [Audio samples](https://shaojinding.github.io/samples/adv).
  *  [Trained model](https://drive.google.com/drive/folders/1FxNC2g8sw9aKQBjL-qBTnkM6VMMUQlLe?usp=sharing).


### Requirements

* Python 3.7 or newer
* PyTorch with CUDA enabled
* TensorFlow 1.13.1
* Run `pip install -r requirements.txt`

### Data preprocessing

We use the speaker encoder model and vocoder model from [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models). We only train the voice conversion model (i.e., synthesizer).

Before running, put the speaker encoder and vocoder at `encoder/saved_models/pretrained.pt` and `vocoder/saved_models/pretrained/pretrained.pt`

1. Download and uncompress [the VCTK dataset](
  https://datashare.is.ed.ac.uk/handle/10283/2651).
2. Manually split the train and test set (there is no official data split). Put them as `<dataset_root>/VCTK/train/p227` and `<dataset_root>/VCTK/test/p228`
3. Run `python synthesizer_preprocess_audio.py <datasets_root>`
4. Run `python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer_train`
5. Run `python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer_test`


## Training and inference

To launch training:

```
$ python synthesizer_train.py vc_adversarial <datasets_root>/SV2TTS/synthesizer_train
```

To run inference, use `synthesis_ppg_script.py`. Change the `syn_dir` to the path of the trained model, e.g., `synthesizer/saved_models/logs-train_adversarial_vctk/taco_pretrained`



# Acknowledgement

The code is adapted from [CorentinJ
/
Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

# Cite the work
```
@article{dingimproving,
  title={Improving the Speaker Identity of Non-Parallel Many-to-Many Voice Conversion with Adversarial Speaker Recognition},
  author={Ding, Shaojin and Zhao, Guanlong and Gutierrez-Osuna, Ricardo}
}
```
