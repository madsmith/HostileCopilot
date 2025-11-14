import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from hostile_copilot.config import OmegaConfig, load_config

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

IS_ROMAN_NUMERAL = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
IS_ALPHA_NUM_CODE = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]+$")
IS_ALPHA_NUM_CODE_SPLIT = re.compile(r"(?<=\d)(?=[A-Z])|(?<=[A-Z])(?=\d)")
IS_ALPHA_UPPER = re.compile(r"^[A-Z]+$")
TOKEN_SEPARATORS = re.compile(r"[-\s,.#()]")

ROMAN_RE = re.compile(r"\b(?:" + "|".join(sorted(ROMAN_MAP.keys(), key=len, reverse=True)) + r")\b")
ENGLISH_RE = re.compile(r"\b(?:" + "|".join(sorted(ENGLISH_MAP.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b(?:" + "|".join(map(str, range(0, 21))) + r")\b")
NUMBER_RE = re.compile(r"\b\d+\b")

def roman_to_int(s: str) -> int:
    roman = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }

    total = 0
    prev_value = 0

    for ch in reversed(s.upper()):
        value = roman[ch]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


#################################################
# Tokenization
#################################################

class Token:
    def __init__(self, value: Any):
        self.value = value
    
    def re_fragment(self) -> str:
        return str(self.value)

    def target_value(self) -> str:
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        return self.value == other.value

class StringToken(Token):
    def __init__(self, value: str):
        super().__init__(value)
    
    def re_fragment(self) -> str:
        return f"(?i:{self.value})"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return f"StringToken({self.value})"

class RomanNumeralToken(Token):
    def __init__(self, value: str):
        super().__init__(value)
        self._int_value = roman_to_int(value)
    
    def re_fragment(self) -> str:
        return f"(?:{self._int_value}|{self.value})"
    
    def target_value(self) -> str:
        return str(self._int_value)
    
    def __str__(self):
        return f"⟦{self.value}⟧"
    
    def __repr__(self):
        return f"RomanNumeralToken({self.value})"

class NumberToken(Token):
    def __init__(self, value: int | str):
        super().__init__(value)
    
    def __str__(self):
        return f"⟦{self.value}⟧"

    def __repr__(self):
        return f"NumberToken({self.value})"

class CodeToken(Token):
    def __init__(self, value: str):
        super().__init__(value)
    
    def re_fragment(self) -> str:
        return f"(?i:{self.value})"
    
    def is_alpha(self) -> bool:
        return self.value.isalpha()
    
    def is_numeric(self) -> bool:
        return self.value.isnumeric()
    
    def __str__(self):
        return f"‹{self.value}›"
    
    def __repr__(self):
        return f"CodeToken({self.value})"

class NormalizedName:
    def __init__(self, name: str):
        self._name = name
        self._tokens: list[Token] = []

        self._tokenize()

    def _tokenize(self):
        tokens = re.split(TOKEN_SEPARATORS, self._name)
        for token in tokens:
            if token == "":
                continue

            if IS_ROMAN_NUMERAL.match(token):
                self._tokens.append(RomanNumeralToken(token))
            
            elif ENGLISH_RE.match(token):
                english_number = token.lower()
                value = ENGLISH_MAP.get(english_number)
                if value is None:
                    logger.warning(f"Unknown English number: {english_number}")
                else:
                    self._tokens.append(NumberToken(value))
            
            elif NUMBER_RE.match(token):
                self._tokens.append(NumberToken(token))
            
            elif IS_ALPHA_NUM_CODE.match(token):
                code_tokens = re.split(IS_ALPHA_NUM_CODE_SPLIT, token)
                for code_token in code_tokens:
                    self._tokens.append(CodeToken(code_token))
            elif IS_ALPHA_UPPER.match(token) and len(token) == 1:
                self._tokens.append(CodeToken(token))
            else:
                lower = token.lower()

                if token in ['&', 'and']:
                    self._tokens.append(StringToken("∧"))
                else:
                    no_apostrophes = re.sub(r"\'", "", lower)
                    self._tokens.append(StringToken(no_apostrophes))

    def matches(self, other: str) -> bool:
        other_name = NormalizedName(other)

        match_target = "".join([token.target_value() for token in other_name._tokens])

        search_re_str = r".*".join([token.re_fragment() for token in self._tokens])
        search_re = re.compile(search_re_str, re.IGNORECASE)

        # logger.debug(f"Searching for {search_re_str} in {match_target}")
        # logger.debug(f"   Tokens: {self._tokens}")
        # logger.debug(f"   Other Tokens: {other_name._tokens}")
        # logger.debug(f"Result: {search_re.search(match_target)}")

        return search_re.search(match_target) is not None

    def target_value(self) -> str:
        return "".join([token.target_value() for token in self._tokens])
    
    def __str__(self):
        return "".join([str(token) for token in self._tokens])

class CanonicalNameProcessor:
    FLAG_MAP = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "VERBOSE": re.VERBOSE,
        "UNICODE": re.UNICODE,
    }

    def __init__(self, config: OmegaConfig | None = None):
        self._config = config
        self._rules: list[dict[str, Any]] = []

        if config is not None:
            self._rules = config.get("location_provider.cleanup_rules", [])

    def process(self, name: str | None) -> str:
        if name is None:
            return ""

        output = self._apply_rules(name)
        return output
        
    def _apply_rules(self, name: str) -> str:
        output = name
        
        for rule in self._rules:
            pattern = rule.get("pattern", None)
            replacement = rule.get("replacement", None)

            if pattern is None:
                logger.warning(f"Invalid rule: {rule} - missing pattern")
                continue

            if replacement is None:
                logger.warning(f"Invalid rule: {rule} - missing replacement")
                continue
            
            flags_list = rule.get("flags", [])
            flags = 0
            for flag in flags_list:
                flags |= getattr(re, flag)
            
            output = re.sub(pattern, replacement, output, flags=flags)
        
        return output
   
def canonical_name(name: str | None) -> str:
    name_lower = name.lower()
    processor = CanonicalNameProcessor()
    return processor.process(name_lower)

if __name__ == "__main__":
    config: OmegaConfig = load_config()

    processor = CanonicalNameProcessor(config)

    # Name normalization
    # Moved to test suite tests/client/components/test_locations.py

    def check_canonical(name: str):
        canonical_name = processor.process(name)
        print(f"\"{name}\" -> \"{canonical_name}\"")

    canonical_names = [
        "Area 18"
    ]

    for name in canonical_names:
        check_canonical(name)