### khmercut

A (fast) Khmer word segmentation toolkit. 

- A single python file
- Using `pycrfsuite` only
- Include Khmer normalize
- CLI Supoprt
- Multiprocess support

```shell
pip install -r requirements.txt
```

### Python

```python
from khmercut import tokenize

tokenize("ឃាត់ខ្លួនជនសង្ស័យ០៤នាក់ ករណីលួចខ្សែភ្លើង នៅស្រុកព្រៃនប់")
# => ['ឃាត់ខ្លួន', 'ជនសង្ស័យ', '០៤', 'នាក់', ' ', 'ករណី', 'លួច', 'ខ្សែភ្លើង', ' ', 'នៅ', 'ស្រុក', 'ព្រៃនប់']
```

### CLI

e.g.

```shell
./khmercut.py large_km.txt --jobs 20 --normalize -d out/ -s "|"
```

Available options

```
usage: khmercut [-h] [-d DIRECTORY] [-s SEPARATOR] [-j JOBS] [-q] [-n] files [files ...]

A fast Khmer word segmentation toolkit.

positional arguments:
  files                 Path to text files

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Output folder
  -s SEPARATOR, --separator SEPARATOR
                        Specify token separator
  -j JOBS, --jobs JOBS  Number of processors
  -q, --quiet           Disable progress output
  -n, --normalize       Normalize input text before processing
```

### Reference

- [Khmer language processing toolkit](https://github.com/VietHoang1512/khmer-nltk)