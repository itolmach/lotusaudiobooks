# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import tempfile
import torch
import sys
import gradio as gr

from huggingface_hub import hf_hub_download

# Setup TTS env
if "vits" not in sys.path:
    sys.path.append("vits")

from vits import commons, utils
from vits.models import SynthesizerTrn


TTS_LANGUAGES = {}
with open(f"data/tts/all_langs.tsv") as f:
    for line in f:
        iso, name = line.split(" ", 1)
        TTS_LANGUAGES[iso] = name


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [
            x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()
        ]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text, lang=None):
        text = self.preprocess_char(text, lang=lang)
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        return txt_filt

    def preprocess_char(self, text, lang=None):
        """
        Special treatement of characters in certain languages
        """
        if lang == "ron":
            text = text.replace("ț", "ţ")
            print(f"{lang} (ț -> ţ): {text}")
        return text


def synthesize(text, lang, speed=None):
    if speed is None:
        speed = 1.0

    lang_code = lang.split()[0].strip()

    vocab_file = hf_hub_download(
        repo_id="facebook/mms-tts",
        filename="vocab.txt",
        subfolder=f"models/{lang_code}",
    )
    config_file = hf_hub_download(
        repo_id="facebook/mms-tts",
        filename="config.json",
        subfolder=f"models/{lang_code}",
    )
    g_pth = hf_hub_download(
        repo_id="facebook/mms-tts",
        filename="G_100000.pth",
        subfolder=f"models/{lang_code}",
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Run inference with {device}")

    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(g_pth, net_g, None)

    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"

    if is_uroman:
        uroman_dir = "uroman"
        assert os.path.exists(uroman_dir)
        uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
        text = text_mapper.uromanize(text, uroman_pl)

    text = text.lower()
    text = text_mapper.filter_oov(text, lang=lang)
    stn_tst = text_mapper.get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0 / speed,
            )[0][0, 0]
            .cpu()
            .float()
            .numpy()
        )

    return gr.Audio.update(value=(hps.data.sampling_rate, hyp)), text


TTS_EXAMPLES = [
    ["I am going to the store.", "eng (English)"],
    ["안녕하세요.", "kor (Korean)"],
    ["क्या मुझे पीने का पानी मिल सकता है?", "hin (Hindi)"],
    ["Tanış olmağıma çox şadam", "azj-script_latin (Azerbaijani, North)"],
    ["Mu zo murna a cikin ƙasar.", "hau (Hausa)"],
]
