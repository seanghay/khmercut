### khmercut

A (fast) Khmer word segmentation toolkit. 

- A single python file
- Using `pycrfsuite` only

```shell
pip install khmercut
```

### Python

```python
from khmercut import tokenize

tokenize("ឃាត់ខ្លួនជនសង្ស័យ០៤នាក់ ករណីលួចខ្សែភ្លើង នៅស្រុកព្រៃនប់")
# => ['ឃាត់ខ្លួន', 'ជនសង្ស័យ', '០៤', 'នាក់', ' ', 'ករណី', 'លួច', 'ខ្សែភ្លើង', ' ', 'នៅ', 'ស្រុក', 'ព្រៃនប់']
```

### Reference

- [Khmer language processing toolkit](https://github.com/VietHoang1512/khmer-nltk)