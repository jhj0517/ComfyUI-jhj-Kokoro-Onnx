import functools
import re
import phonemizer


@functools.lru_cache
def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts


def normalize_text(text):
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', chr(8220)).replace('»', chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace('(', '«').replace(')', '»')
    for a, b in zip('、。！，：；？', ',.!,:;?'):
        text = text.replace(a, b+' ')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text)
    text = re.sub(r'(?<=\d)S', ' S', text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", 's', text)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)
    return text.strip()


def phonemize(text, lang, norm=True):
    if norm:
        text = normalize_text(text)

    phonemizers = dict(
        a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
        b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
    )

    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    # https://en.wiktionary.org/wiki/kokoro#English
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', 'z', ps)
    if lang == 'a':
        ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()