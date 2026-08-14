### khmercut

A (fast) Khmer word segmentation toolkit.

```shell
pip install khmercut
```

### Python

```python
from khmercut import tokenize

tokenize("ឃាត់ខ្លួនជនសង្ស័យ០៤នាក់ ករណីលួចខ្សែភ្លើង នៅស្រុកព្រៃនប់")
# => ['ឃាត់ខ្លួន', 'ជនសង្ស័យ', '០៤', 'នាក់', ' ', 'ករណី', 'លួច', 'ខ្សែភ្លើង', ' ', 'នៅ', 'ស្រុក', 'ព្រៃនប់']
```

### Neural model

A distilled neural segmenter. More accurate, and it can split compound words.

```shell
pip install khmercut[nn]
```

```python
from khmercut.nn import tokenize

tokenize("ឃាត់ខ្លួនជនសង្ស័យ០៤នាក់")
# => ['ឃាត់ខ្លួន', 'ជនសង្ស័យ', '០៤', 'នាក់']

tokenize("ឃាត់ខ្លួនជនសង្ស័យ០៤នាក់", deep=True)
# => [['ឃាត់', 'ខ្លួន'], ['ជន', 'សង្ស័យ'], ['០៤'], ['នាក់']]
```

With `deep=True` each word becomes a list of its sub-words. Spaces are returned
as `' '` and newlines as `'\n'`.

The model is 2.2 MB (int8 ONNX, 320 hidden units). On the Khmer ALT gold corpus
it scores 90.6 F1 on sub-words and 74.7 F1 on words.

Both backends load lazily, so importing one does not pull in the other.

### Reference

- [Khmer language processing toolkit](https://github.com/VietHoang1512/khmer-nltk)
- [Asian Language Treebank](http://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html)
