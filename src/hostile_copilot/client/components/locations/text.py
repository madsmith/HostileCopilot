import re

ROMAN_MAP = {
    "I":            1,
    "II":           2,
    "III":          3,
    "IV":           4,
    "V":            5,
    "VI":           6,
    "VII":          7,
    "VIII":         8,
    "IX":           9,
    "X":            10,
    "XI":           11,
    "XII":          12,
    "XIII":         13,
    "XIV":          14,
    "XV":           15,
    "XVI":          16,
    "XVII":         17,
    "XVIII":        18,
    "XIX":          19,
    "XX":           20,
}

ENGLISH_MAP = {
    "zero":         0,
    "one":          1,
    "two":          2,
    "three":        3,
    "four":         4,
    "five":         5,
    "six":          6,
    "seven":        7,
    "eight":        8,
    "nine":         9,
    "ten":          10,
    "eleven":       11,
    "twelve":       12,
    "thirteen":     13,
    "fourteen":     14,
    "fifteen":      15,
    "sixteen":      16,
    "seventeen":    17,
    "eighteen":     18,
    "nineteen":     19,
    "twenty":       20,
}

ROMAN_RE = re.compile(r"\b(?:" + "|".join(sorted(ROMAN_MAP.keys(), key=len, reverse=True)) + r")\b")
ENGLISH_RE = re.compile(r"\b(?:" + "|".join(sorted(ENGLISH_MAP.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b(?:" + "|".join(map(str, range(0, 21))) + r")\b")

def _fn_replace_roman_numeral(match: re.Match[str]) -> str:
    roman_numeral = match.group(0).upper()
    replacement = ROMAN_MAP.get(roman_numeral, roman_numeral)
    return f" ⟦{replacement}⟧ "

def _fn_replace_english_numeral(match: re.Match[str]) -> str:
    english_number = match.group(0)
    lower_english_number = english_number.lower()
    replacement = ENGLISH_MAP.get(lower_english_number, lower_english_number)
    return f" ⟦{replacement}⟧ "

def _fn_replace_number(match: re.Match[str]) -> str:
    number = match.group(0)
    return f" ⟦{number}⟧ "
    
def _separate_letter_digits(str: str) -> str:
    left_to_right = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", str)
    right_to_left = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", left_to_right)
    return right_to_left

def normalize_name(name: str | None) -> str:
    if name is None:
        return ""
    
    letter_digit_separated = _separate_letter_digits(name)
    
    dehyphenated = re.sub(r"-", " ", letter_digit_separated)

    # Normalize numbers into distinct tokens
    number_normalized = NUMBER_RE.sub(_fn_replace_number, dehyphenated)
    roman_normalized = ROMAN_RE.sub(_fn_replace_roman_numeral, number_normalized)
    english_normalized = ENGLISH_RE.sub(_fn_replace_english_numeral, roman_normalized)

    # Remove redundant spaces and boundary separators like '-'
    boundary_separated = re.sub(r"-\s*⟦", " ⟦", english_normalized)
    deapostrophized = re.sub(r"\'", "", boundary_separated)
    conjunction_spelled_out = re.sub(r"\b&\b", "and", deapostrophized)
    space_consolidated = re.sub(r"\s+", " ", conjunction_spelled_out)
    
    return space_consolidated.strip().lower()



if __name__ == "__main__":
    def check(name: str):
        print(f"\"{name}\" -> \"{normalize_name(name)}\"")
    
    names = [
        "The Moon",
        "The Moon X",
        "The Moon-XIX",
        "The Moon One",
        "The Moon One-XIX",
        "The Moon four",
        "ARC-L1",
        "RAB-ION",
        "RUPTURA PAF-II",
        "Operations Depot Lyria-1",
        "Greycat Stanton IV Production Complex",
        "The Moon-1",
        "Shubin Processing Facility SPAL-3",
        "SAL-2",
        "Arccorp Mining Area 141",
        "Shubin Mining Facility SM0-10",
        "Shubin Mining Facility SM0-",
        "Farro Datacenter X",
        "Lazarus Transport Hub Tithonus-III",
        "Pyro IV", "Pyro 4",
        None,
        "",
    ]

    for name in names:
        check(name)