from pycrfsuite import Tagger
import os

SEPARATOR = "\u200b"
KHCONST = set("កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ")
KHVOWEL = set("឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8")
KHSUB = set("្")
KHDIAC = set("\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0")
KHSYM = set("៕។៛ៗ៚៙៘,.? ")
KHNUMBER = set("០១២៣៤៥៦៧៨៩0123456789")
KHLUNAR = set("᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿")
NS = "NS"

model_path =  os.path.join(os.path.dirname(__file__), "crf_ner_10000.crfsuite")
tagger = Tagger()
tagger.open(model_path)

def is_khmer_char(ch: str):
    if (ch >= "\u1780") and (ch <= "\u17ff"):
        return True
    if ch in KHSYM:
        return True
    if ch in KHLUNAR:
        return True
    return False


def is_start_of_kcc(ch: str):
    if is_khmer_char(ch):
        if ch in KHCONST:
            return True
        if ch in KHSYM:
            return True
        if ch in KHNUMBER:
            return True
        if ch in KHLUNAR:
            return True
        return False
    return True


def seg_kcc(word: str):
    segs = []
    cur = ""
    for i, c in enumerate(word):
        cur += c
        nextchar = word[i + 1] if (i + 1 < len(word)) else ""
        if (
            not is_khmer_char(c)
            and nextchar != " "
            and nextchar != ""
            and not is_khmer_char(nextchar)
        ):
            continue

        if c in KHNUMBER and nextchar in KHNUMBER:
            continue

        if not is_khmer_char(c) or nextchar == " " or nextchar == "":
            segs.append(cur)
            cur = ""
        elif is_start_of_kcc(nextchar) and not (c in KHSUB):
            segs.append(cur)
            cur = ""

    return segs


def get_type(char: str):
    if char in KHCONST:
        return "C"

    if char in KHVOWEL:
        return "W"

    if char in KHSUB:
        return "S"

    if char in KHDIAC:
        return "D"

    return NS


def is_no_space(k):
    if get_type(k[0]) == NS:
        return True
    return False


def kcc_type(k):
    if len(k) == 1:
        return get_type(k)
    else:
        return "K" + str(len(k))


def kcc_to_features(kccs, i):
    maxi = len(kccs)
    kcc = kccs[i]
    features = {"kcc": kcc, "t": kcc_type(kcc), "ns": is_no_space(kcc)}

    if i >= 1:
        features.update(
            {
                "kcc[-1]": kccs[i - 1],
                "kcc[-1]t": kcc_type(kccs[i - 1]),
                "kcc[-1:0]": kccs[i - 1] + kccs[i],
                "ns-1": is_no_space(kccs[i - 1]),
            }
        )
    else:
        features["BOS"] = True

    if i >= 2:
        features.update(
            {
                "kcc[-2]": kccs[i - 2],
                "kcc[-2]t": kcc_type(kccs[i - 2]),
                "kcc[-2:-1]": kccs[i - 2] + kccs[i - 1],
                "kcc[-2:0]": kccs[i - 2] + kccs[i - 1] + kccs[i],
            }
        )
    if i >= 3:
        features.update(
            {
                "kcc[-3]": kccs[i - 3],
                "kcc[-3]t": kcc_type(kccs[i - 3]),
                "kcc[-3:0]": kccs[i - 3] + kccs[i - 2] + kccs[i - 1] + kccs[i],
                "kcc[-3:-1]": kccs[i - 3] + kccs[i - 2] + kccs[i - 1],
                "kcc[-3:-2]": kccs[i - 3] + kccs[i - 2],
            }
        )

    if i < maxi - 1:
        features.update(
            {
                "kcc[+1]": kccs[i + 1],
                "kcc[+1]t": kcc_type(kccs[i + 1]),
                "kcc[+1:0]": kccs[i] + kccs[i + 1],
                "ns+1": is_no_space(kccs[i + 1]),
            }
        )
    else:
        features["EOS"] = True

    if i < maxi - 2:
        features.update(
            {
                "kcc[+2]": kccs[i + 2],
                "kcc[+2]t": kcc_type(kccs[i + 2]),
                "kcc[+1:+2]": kccs[i + 1] + kccs[i + 2],
                "kcc[0:+2]": kccs[i + 0] + kccs[i + 1] + kccs[i + 2],
                "ns+2": is_no_space(kccs[i + 2]),
            }
        )
    if i < maxi - 3:
        features.update(
            {
                "kcc[+3]": kccs[i + 3],
                "kcc[+3]t": kcc_type(kccs[i + 3]),
                "kcc[+2:+3]": kccs[i + 2] + kccs[i + 3],
                "kcc[+1:+3]": kccs[i + 1] + kccs[i + 2] + kccs[i + 3],
                "kcc[0:+3]": kccs[i + 0] + kccs[i + 1] + kccs[i + 2] + kccs[i + 3],
            }
        )

    return features


def create_kcc_features(kccs):
    return [kcc_to_features(kccs, i) for i in range(len(kccs))]


def tokenize(text):
    graphemes = seg_kcc(text)
    features = create_kcc_features(graphemes)

    tokens = []
    for i, tag in enumerate(tagger.tag(features)):
        if tag == "1" or i == 0:
            tokens.append(graphemes[i])
        else:
            tokens[-1] += graphemes[i]

    return tokens
