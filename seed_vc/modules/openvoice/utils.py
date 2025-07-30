# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Original work Copyright (C) 2025 Plachtaa <https://github.com/Plachtaa>
# Original source: https://github.com/Plachtaa/seed-vc

"""Utility functions for OpenVoice."""

import json
import re
from typing import Any, List

import numpy as np


def get_hparams_from_file(config_path: str) -> "HParams":
    """Load hyperparameters from a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        HParams object containing the configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams:
    """Hyperparameters container with attribute-style access.

    Provides both dictionary and attribute access to hyperparameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HParams with keyword arguments.

        Args:
            **kwargs: Hyperparameter key-value pairs.
        """
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self) -> Any:
        """Return keys of hyperparameters."""
        return self.__dict__.keys()

    def items(self) -> Any:
        """Return items of hyperparameters."""
        return self.__dict__.items()

    def values(self) -> Any:
        """Return values of hyperparameters."""
        return self.__dict__.values()

    def __len__(self) -> int:
        """Return number of hyperparameters."""
        return len(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        """Get hyperparameter by key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set hyperparameter by key."""
        return setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in hyperparameters."""
        return key in self.__dict__

    def __repr__(self) -> str:
        """Return string representation of hyperparameters."""
        return self.__dict__.__repr__()


def string_to_bits(string: str, pad_len: int = 8) -> np.ndarray:
    """Convert string to binary bit array.

    Args:
        string: Input string to convert.
        pad_len: Length to pad the output array.

    Returns:
        NumPy array of bits with shape (pad_len, 8).
    """
    # Convert each character to its ASCII value
    ascii_values = [ord(char) for char in string]

    # Convert ASCII values to binary representation
    binary_values = [bin(value)[2:].zfill(8) for value in ascii_values]

    # Convert binary strings to integer arrays
    bit_arrays = [[int(bit) for bit in binary] for binary in binary_values]

    # Convert list of arrays to NumPy array
    numpy_array = np.array(bit_arrays)
    numpy_array_full = np.zeros((pad_len, 8), dtype=numpy_array.dtype)
    numpy_array_full[:, 2] = 1
    max_len = min(pad_len, len(numpy_array))
    numpy_array_full[:max_len] = numpy_array[:max_len]
    return numpy_array_full


def bits_to_string(bits_array: np.ndarray) -> str:
    """Convert binary bit array back to string.

    Args:
        bits_array: NumPy array of bits.

    Returns:
        Decoded string.
    """
    # Convert each row of the array to a binary string
    binary_values = ["".join(str(bit) for bit in row) for row in bits_array]

    # Convert binary strings to ASCII values
    ascii_values = [int(binary, 2) for binary in binary_values]

    # Convert ASCII values to characters
    output_string = "".join(chr(value) for value in ascii_values)

    return output_string


def split_sentence(text: str, min_len: int = 10, language_str: str = "[EN]") -> List[str]:
    """Split text into sentences based on language.

    Args:
        text: Input text to split.
        min_len: Minimum length for sentence splitting.
        language_str: Language identifier ("[EN]" for English, others for Chinese).

    Returns:
        List of split sentences.
    """
    if language_str in ["EN"]:
        sentences = split_sentences_latin(text, min_len=min_len)
    else:
        sentences = split_sentences_zh(text, min_len=min_len)
    return sentences


def split_sentences_latin(text: str, min_len: int = 10) -> List[str]:
    """Split long Latin-script sentences into list of shorter ones.

    Args:
        text: Input text with Latin script.
        min_len: Minimum word count for splitting.

    Returns:
        List of output sentences.
    """
    # deal with dirty sentences
    text = re.sub("[。！？；]", ".", text)
    text = re.sub("[，]", ",", text)
    text = re.sub("[“”]", '"', text)
    text = re.sub("[‘’]", "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    text = re.sub("[\n\t ]+", " ", text)
    text = re.sub("([,.!?;])", r"\1 $#!", text)
    # split
    sentences = [s.strip() for s in text.split("$#!")]
    if len(sentences[-1]) == 0:
        del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        # print(sent)
        new_sent.append(sent)
        count_len += len(sent.split(" "))
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(" ".join(new_sent))
            new_sent = []
    return merge_short_sentences_latin(new_sentences)


def merge_short_sentences_latin(sens: List[str]) -> List[str]:
    """Avoid short sentences by merging them with the following sentence.

    Args:
        sens: List of input sentences.

    Returns:
        List of merged output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentence is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except Exception:
        pass
    return sens_out


def split_sentences_zh(text: str, min_len: int = 10) -> List[str]:
    """Split long Chinese sentences into list of shorter ones.

    Args:
        text: Input text with Chinese characters.
        min_len: Minimum character count for splitting.

    Returns:
        List of output sentences.
    """
    text = re.sub("[。！？；]", ".", text)
    text = re.sub("[，]", ",", text)
    # 将文本中的换行符、空格和制表符替换为空格
    text = re.sub("[\n\t ]+", " ", text)
    # 在标点符号后添加一个空格
    text = re.sub("([,.!?;])", r"\1 $#!", text)
    # 分隔句子并去除前后空格
    # sentences = [s.strip() for s in re.split('(。|！|？|；)', text)]
    sentences = [s.strip() for s in text.split("$#!")]
    if len(sentences[-1]) == 0:
        del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(" ".join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def merge_short_sentences_zh(sens: List[str]) -> List[str]:
    """Avoid short Chinese sentences by merging them with the following sentence.

    Args:
        sens: List of input sentences.

    Returns:
        List of merged output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except Exception:
        pass
    return sens_out
