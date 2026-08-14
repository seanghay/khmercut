import json
import os
import re

import numpy as np
import onnxruntime as ort

MODEL_DIR = os.path.join(os.path.dirname(__file__), "nn_model")
LABELS = ("0", "^", "_", "}{", "~")
ZWSP = "​"
PUNCT = set("។៕៖៘៙៚ៜ")
TOKENS = re.compile(r"([ក-៿ ]+)|([^ក-៿ ]+)")
SPACES = re.compile("[   -   　﻿]")
MAX_LEN = 512
OVERLAP = 32

_model = None


def char_class(ch):
    o = ord(ch)
    if 0x1780 <= o <= 0x17A2:
        return "C"
    if 0x17A3 <= o <= 0x17B3:
        return "IV"
    if 0x17B6 <= o <= 0x17C8:
        return "V"
    if 0x17C9 <= o <= 0x17D1:
        return "US"
    if ch in PUNCT:
        return "END"
    if 0x19E0 <= o <= 0x19FF:
        return "LN"
    if 0x17F0 <= o <= 0x17F9:
        return "AN"
    if o == 0x17D2:
        return "SUB"
    return "NS"


def encode(text):
    for match in TOKENS.finditer(text):
        if match.group(2) is not None:
            yield match.group(2), "NS"
            continue
        for ch in match.group(1):
            if ch == " ":
                yield ZWSP, "ZS"
            else:
                yield ch, char_class(ch)


def viterbi(emissions, transitions, start, end):
    score = start + emissions[0]
    backptr = np.empty(emissions.shape, dtype=np.int32)
    for t in range(1, len(emissions)):
        cand = score[:, None] + transitions
        best = cand.argmax(axis=0)
        backptr[t] = best
        score = cand[best, np.arange(len(best))] + emissions[t]
    path = [int((score + end).argmax())]
    for t in range(len(emissions) - 1, 0, -1):
        path.append(int(backptr[t][path[-1]]))
    path.reverse()
    return path


def decode(values, labels, deep):
    chunk = ""
    word = []
    for value, label in zip(values, labels):
        if value == ZWSP:
            if chunk:
                word.append(chunk)
                chunk = ""
            if word:
                yield word if deep else "".join(word)
                word = []
            yield [" "] if deep else " "
            continue
        chunk += value
        if label == "0":
            continue
        if label != "}{":
            if deep:
                word.append(chunk)
                chunk = ""
            continue
        word.append(chunk)
        chunk = ""
        yield word if deep else "".join(word)
        word = []
    if chunk:
        word.append(chunk)
    if word:
        yield word if deep else "".join(word)


class Model:
    def __init__(self, model_dir=MODEL_DIR):
        with open(os.path.join(model_dir, "vocab.json")) as f:
            vocab = json.load(f)
        self.stoi = {v: i for i, v in enumerate(vocab["itos"])}
        self.class_id = {c: i for i, c in enumerate(vocab["classes"])}
        crf = np.load(os.path.join(model_dir, "crf.npz"))
        self.transitions = crf["transitions"].astype(np.float32)
        self.start = crf["start_transitions"].astype(np.float32)
        self.end = crf["end_transitions"].astype(np.float32)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.int8.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

    def bucket(self, value):
        if value.isdigit():
            shape = "digit"
        elif value.isascii() and value.isalpha():
            shape = "latin"
        elif any(c.isalnum() for c in value):
            shape = "mixed"
        else:
            shape = "other"
        n = len(value)
        size = "1" if n == 1 else "2_4" if n <= 4 else "5plus"
        return self.stoi.get("<ns:%s:%s>" % (shape, size), 1)

    def emissions(self, ids, classes):
        return self.session.run(
            None,
            {
                "ids": np.asarray([ids], dtype=np.int64),
                "cls": np.asarray([classes], dtype=np.int64),
            },
        )[0][0]

    def labels(self, pairs):
        ids, classes = [], []
        for value, name in pairs:
            index = self.stoi.get(value)
            ids.append(self.bucket(value) if index is None else index)
            classes.append(self.class_id[name])

        n = len(ids)
        if n == 0:
            return []
        if n <= MAX_LEN:
            return viterbi(
                self.emissions(ids, classes), self.transitions, self.start, self.end
            )

        out = [0] * n
        stride = MAX_LEN - 2 * OVERLAP
        start = 0
        while start < n:
            stop = min(start + MAX_LEN, n)
            path = viterbi(
                self.emissions(ids[start:stop], classes[start:stop]),
                self.transitions,
                self.start,
                self.end,
            )
            head = 0 if start == 0 else OVERLAP
            tail = (stop - start) if stop == n else (stop - start - OVERLAP)
            out[start + head : start + tail] = path[head:tail]
            if stop == n:
                break
            start += stride
        return out


def load(model_dir=MODEL_DIR):
    global _model
    if _model is None:
        _model = Model(model_dir)
    return _model


def tokenize(text, deep=False):
    model = load()
    tokens = []
    lines = SPACES.sub(" ", text.replace(ZWSP, "")).strip().split("\n")
    for i, line in enumerate(lines):
        pairs = list(encode(line))
        labels = [LABELS[k] for k in model.labels(pairs)]
        tokens.extend(decode([v for v, _ in pairs], labels, deep))
        if i < len(lines) - 1:
            tokens.append(["\n"] if deep else "\n")
    return tokens
