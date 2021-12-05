import re
from unidecode import unidecode
from pykakasi import kakasi
from jamo import hangul_to_jamo
from .numbers import normalize_numbers


_kks = kakasi()
_kks.setMode('H', 'a')
_kks.setMode('K', 'a')
_kks.setMode('J', 'a')
_kks.setMode('E', 'a')
_kks.setMode('s', True)
_conv = _kks.getConverter()

#White space
_whitespace_re = re.compile(r'\s+')

#訳語処理のためのリスト
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

def replace_abbreviations(text):
    for abbr, replacement in _abbreviations:
        text = re.sub(abbr, replacement, text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unidecode(text)

def english_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = replace_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def japanese_cleaners(text):
    text = _conv.do(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = collapse_whitespace(text)
    return text

def korean_cleaners(text):
    text = ''.join(list(hangul_to_jamo(text)))
    return text

def transliteration_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text