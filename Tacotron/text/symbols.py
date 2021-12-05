from jamo import hangul_to_jamo

_pad        = '_'
_EOS        = '~'
_punctuation = '!\'(),.:;? '
_SPACE = ' '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_jamo_leads = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_jamo_vowels = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_jamo_tails = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])
_valid_chars = _jamo_leads + _jamo_vowels + _jamo_tails

def get_symbols(mod):
    if mod == 'english':
        symbols = [_pad] + list(_EOS) + list(_special) + list(_punctuation) + list(_letters)
        print('symbols = ', len(symbols))
    elif mod == 'korean':
        symbols = [_pad] + list(_EOS) + list(_SPACE) + list(_valid_chars)
        print('symbols = ', len(symbols))
    elif mod == 'japanese':
        symbols = symbols = [_pad] + list(_EOS) + list(_special) + list(_punctuation) + list(_letters)
    
    return symbols
