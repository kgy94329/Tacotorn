import re
from text import text_cleaner
from text.symbols import get_symbols
from jamo import hangul_to_jamo
import sys
sys.path.append('..')
from util.hparams import HyperParams as hp

_symbol_to_id = {s: i for i, s in enumerate(get_symbols(hp.mod))}
_id_to_symbol = {i: s for i, s in enumerate(get_symbols(hp.mod))}

def text_to_sequence(text, cleaner_names):
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    if hp.mod == 'korean' and not 0x1100 <= ord(text[0]) <= 0x1113:
      text = ''.join(list(hangul_to_jamo(text)))
    sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
    sequence.append(_symbol_to_id['~'])
    return sequence


def sequence_to_text(sequence):
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(text_cleaner, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'
