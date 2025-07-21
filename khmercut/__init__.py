from pycrfsuite import Tagger
import os
from typing import List, Dict, Tuple
import array

# Ultra-fast character lookup tables using arrays for O(1) access
_CHAR_TYPE_LOOKUP = array.array('b', [0] * 65536)  # byte array for Unicode BMP
_IS_KHMER_LOOKUP = array.array('b', [0] * 65536)
_IS_START_KCC_LOOKUP = array.array('b', [0] * 65536)

# Constants for character types (using integers for speed)
TYPE_C = 1  # Consonant
TYPE_W = 2  # Vowel
TYPE_S = 3  # Subscript
TYPE_D = 4  # Diacritic
TYPE_NS = 0  # Non-special

# Unicode ranges as integers for ultra-fast comparison
KHMER_START = 0x1780
KHMER_END = 0x17ff

# Character sets as frozensets for fast membership testing
KHCONST = frozenset("កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ")
KHVOWEL = frozenset("឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8")
KHSUB = frozenset("្")
KHDIAC = frozenset("\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0")
KHSYM = frozenset("៕។៛ៗ៚៙៘,.? ")
KHNUMBER = frozenset("០១២៣៤៥៦៧៨៩0123456789")
KHLUNAR = frozenset("᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿")

# Pre-compute all character classifications at module load time


def _init_lookup_tables():
    """Initialize ultra-fast lookup tables"""
    # Mark all Khmer characters
    for i in range(KHMER_START, KHMER_END + 1):
        _IS_KHMER_LOOKUP[i] = 1

    # Mark specific character types
    for char in KHCONST:
        code = ord(char)
        _CHAR_TYPE_LOOKUP[code] = TYPE_C
        _IS_KHMER_LOOKUP[code] = 1
        _IS_START_KCC_LOOKUP[code] = 1

    for char in KHVOWEL:
        code = ord(char)
        _CHAR_TYPE_LOOKUP[code] = TYPE_W
        _IS_KHMER_LOOKUP[code] = 1

    for char in KHSUB:
        code = ord(char)
        _CHAR_TYPE_LOOKUP[code] = TYPE_S
        _IS_KHMER_LOOKUP[code] = 1

    for char in KHDIAC:
        code = ord(char)
        _CHAR_TYPE_LOOKUP[code] = TYPE_D
        _IS_KHMER_LOOKUP[code] = 1

    for char in KHSYM:
        code = ord(char)
        _IS_KHMER_LOOKUP[code] = 1
        _IS_START_KCC_LOOKUP[code] = 1

    for char in KHNUMBER:
        code = ord(char)
        _IS_KHMER_LOOKUP[code] = 1
        _IS_START_KCC_LOOKUP[code] = 1

    for char in KHLUNAR:
        code = ord(char)
        _IS_KHMER_LOOKUP[code] = 1
        _IS_START_KCC_LOOKUP[code] = 1

    # Mark non-Khmer characters as start of KCC
    for i in range(65536):
        if not _IS_KHMER_LOOKUP[i]:
            _IS_START_KCC_LOOKUP[i] = 1


# Initialize lookup tables immediately
_init_lookup_tables()

# Ultra-fast caches using dictionaries with pre-allocated size
_KCC_TYPE_CACHE = {}
_FEATURE_CACHE = {}

# Pre-populate KCC type cache with common patterns
_KCC_TYPE_CACHE.update({
    'C': 'C', 'W': 'W', 'S': 'S', 'D': 'D',
    'K2': 'K2', 'K3': 'K3', 'K4': 'K4', 'K5': 'K5'
})

# String constants for feature keys (avoid string allocation)
KCC_KEY = "kcc"
T_KEY = "t"
NS_KEY = "ns"
BOS_KEY = "BOS"
EOS_KEY = "EOS"

# Initialize tagger
model_path = os.path.join(os.path.dirname(__file__), "crf_ner_10000.crfsuite")
tagger = Tagger()
tagger.open(model_path)

# Ultra-fast inline functions using lookup tables


def is_khmer_char_fast(char_code: int) -> bool:
    """Ultra-fast Khmer character detection using array lookup"""
    return bool(_IS_KHMER_LOOKUP[char_code])


def is_start_of_kcc_fast(char_code: int) -> bool:
    """Ultra-fast start of KCC detection using array lookup"""
    return bool(_IS_START_KCC_LOOKUP[char_code])


def get_type_fast(char_code: int) -> int:
    """Ultra-fast character type lookup using array"""
    return _CHAR_TYPE_LOOKUP[char_code]


def seg_kcc_ultra_fast(text: str) -> List[str]:
    """Ultra-fast KCC segmentation using character codes"""
    if not text:
        return []

    segments = []
    start_idx = 0
    text_len = len(text)

    i = 0
    while i < text_len:
        char_code = ord(text[i])

        # Check if we should segment at this position
        should_segment = False

        if i == text_len - 1:
            # Last character
            should_segment = True
        else:
            next_char_code = ord(text[i + 1])

            # Fast path: non-Khmer followed by non-Khmer and not space
            if (not _IS_KHMER_LOOKUP[char_code] and
                next_char_code != 32 and  # space
                    not _IS_KHMER_LOOKUP[next_char_code]):
                i += 1
                continue

            # Number sequence check
            if (char_code in {0x17E0, 0x17E1, 0x17E2, 0x17E3, 0x17E4, 0x17E5, 0x17E6, 0x17E7, 0x17E8, 0x17E9,  # Khmer digits
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57} and  # ASCII digits
                next_char_code in {0x17E0, 0x17E1, 0x17E2, 0x17E3, 0x17E4, 0x17E5, 0x17E6, 0x17E7, 0x17E8, 0x17E9,
                                   48, 49, 50, 51, 52, 53, 54, 55, 56, 57}):
                i += 1
                continue

            # Segment conditions
            if (not _IS_KHMER_LOOKUP[char_code] or
                next_char_code == 32 or  # space
                    # subscript
                    (_IS_START_KCC_LOOKUP[next_char_code] and char_code != 0x17D2)):
                should_segment = True

        if should_segment:
            segments.append(text[start_idx:i + 1])
            start_idx = i + 1

        i += 1

    return segments


def kcc_type_ultra_fast(kcc: str) -> str:
    """Ultra-fast KCC type determination with caching"""
    if kcc in _KCC_TYPE_CACHE:
        return _KCC_TYPE_CACHE[kcc]

    kcc_len = len(kcc)
    if kcc_len == 1:
        char_type = _CHAR_TYPE_LOOKUP[ord(kcc)]
        if char_type == TYPE_C:
            result = "C"
        elif char_type == TYPE_W:
            result = "W"
        elif char_type == TYPE_S:
            result = "S"
        elif char_type == TYPE_D:
            result = "D"
        else:
            result = "NS"
    else:
        result = f"K{kcc_len}"

    _KCC_TYPE_CACHE[kcc] = result
    return result


def create_features_ultra_fast(kccs: List[str]) -> List[Dict]:
    """Ultra-fast feature creation with aggressive optimization"""
    if not kccs:
        return []

    features_list = []
    kccs_len = len(kccs)

    # Pre-compute all KCC types to avoid repeated calculations
    kcc_types = [kcc_type_ultra_fast(kcc) for kcc in kccs]

    for i in range(kccs_len):
        kcc = kccs[i]
        kcc_type_current = kcc_types[i]

        # Create base features
        features = {
            KCC_KEY: kcc,
            T_KEY: kcc_type_current,
            NS_KEY: _CHAR_TYPE_LOOKUP[ord(kcc[0])] == TYPE_NS
        }

        # Backward features
        if i >= 1:
            prev_kcc = kccs[i - 1]
            features.update({
                "kcc[-1]": prev_kcc,
                "kcc[-1]t": kcc_types[i - 1],
                "kcc[-1:0]": prev_kcc + kcc,
                "ns-1": _CHAR_TYPE_LOOKUP[ord(prev_kcc[0])] == TYPE_NS,
            })
        else:
            features[BOS_KEY] = True

        if i >= 2:
            prev2_kcc = kccs[i - 2]
            prev_kcc = kccs[i - 1]
            features.update({
                "kcc[-2]": prev2_kcc,
                "kcc[-2]t": kcc_types[i - 2],
                "kcc[-2:-1]": prev2_kcc + prev_kcc,
                "kcc[-2:0]": prev2_kcc + prev_kcc + kcc,
            })

        if i >= 3:
            prev3_kcc = kccs[i - 3]
            prev2_kcc = kccs[i - 2]
            prev_kcc = kccs[i - 1]
            features.update({
                "kcc[-3]": prev3_kcc,
                "kcc[-3]t": kcc_types[i - 3],
                "kcc[-3:0]": prev3_kcc + prev2_kcc + prev_kcc + kcc,
                "kcc[-3:-1]": prev3_kcc + prev2_kcc + prev_kcc,
                "kcc[-3:-2]": prev3_kcc + prev2_kcc,
            })

        # Forward features
        if i < kccs_len - 1:
            next_kcc = kccs[i + 1]
            features.update({
                "kcc[+1]": next_kcc,
                "kcc[+1]t": kcc_types[i + 1],
                "kcc[+1:0]": kcc + next_kcc,
                "ns+1": _CHAR_TYPE_LOOKUP[ord(next_kcc[0])] == TYPE_NS,
            })
        else:
            features[EOS_KEY] = True

        if i < kccs_len - 2:
            next_kcc = kccs[i + 1]
            next2_kcc = kccs[i + 2]
            features.update({
                "kcc[+2]": next2_kcc,
                "kcc[+2]t": kcc_types[i + 2],
                "kcc[+1:+2]": next_kcc + next2_kcc,
                "kcc[0:+2]": kcc + next_kcc + next2_kcc,
                "ns+2": _CHAR_TYPE_LOOKUP[ord(next2_kcc[0])] == TYPE_NS,
            })

        if i < kccs_len - 3:
            next_kcc = kccs[i + 1]
            next2_kcc = kccs[i + 2]
            next3_kcc = kccs[i + 3]
            features.update({
                "kcc[+3]": next3_kcc,
                "kcc[+3]t": kcc_types[i + 3],
                "kcc[+2:+3]": next2_kcc + next3_kcc,
                "kcc[+1:+3]": next_kcc + next2_kcc + next3_kcc,
                "kcc[0:+3]": kcc + next_kcc + next2_kcc + next3_kcc,
            })

        features_list.append(features)

    return features_list


def tokenize_ultra_fast(text: str) -> List[str]:
    """Ultra-fast tokenization with maximum optimization"""
    if not text:
        return []

    # Step 1: Ultra-fast segmentation
    graphemes = seg_kcc_ultra_fast(text)
    if not graphemes:
        return []

    # Step 2: Ultra-fast feature creation
    features = create_features_ultra_fast(graphemes)

    # Step 3: CRF tagging
    tags = tagger.tag(features)

    # Step 4: Ultra-fast token assembly
    tokens = []
    current_parts = []

    for i, (grapheme, tag) in enumerate(zip(graphemes, tags)):
        if tag == "1" or i == 0:
            if current_parts:
                tokens.append(''.join(current_parts))
                current_parts = [grapheme]
            else:
                current_parts.append(grapheme)
        else:
            current_parts.append(grapheme)

    if current_parts:
        tokens.append(''.join(current_parts))

    return tokens

# Batch processing for maximum throughput


def tokenize_batch_ultra_fast(texts: List[str]) -> List[List[str]]:
    """Ultra-fast batch tokenization"""
    return [tokenize_ultra_fast(text) for text in texts]

# Memory management


def clear_caches_ultra_fast():
    """Clear caches and reinitialize for maximum performance"""
    global _KCC_TYPE_CACHE, _FEATURE_CACHE
    _KCC_TYPE_CACHE.clear()
    _FEATURE_CACHE.clear()

    # Re-populate essential cache entries
    _KCC_TYPE_CACHE.update({
        'C': 'C', 'W': 'W', 'S': 'S', 'D': 'D',
        'K2': 'K2', 'K3': 'K3', 'K4': 'K4', 'K5': 'K5'
    })


# Compatibility aliases
tokenize = tokenize_ultra_fast
seg_kcc = seg_kcc_ultra_fast
create_kcc_features = create_features_ultra_fast
