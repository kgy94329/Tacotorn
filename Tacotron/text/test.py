from pykakasi import kakasi

text = input()

_kks = kakasi()
_kks.setMode('H', 'a')
_kks.setMode('K', 'a')
_kks.setMode('J', 'a')
_kks.setMode('E', 'a')
_kks.setMode('s', True)
_conv = _kks.getConverter()

print(_conv.do(text))