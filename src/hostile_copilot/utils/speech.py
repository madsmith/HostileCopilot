"""
Utility functions for converted text to speech (TTS)
"""
import re

def roundify_numbers(text: str | int | float) -> str:
    def humanize(num: float) -> str:
        if num >= 1_000_000_000:  # Billions
            billions = num / 1_000_000_000
            if billions < 10:
                return f"{billions:.1f}".rstrip('0').rstrip('.') + " billion"
            else:
                return f"{int(round(billions))} billion"
        elif num >= 1_000_000:    # Millions
            millions = num / 1_000_000
            if millions < 10:
                return f"{millions:.1f}".rstrip('0').rstrip('.') + " million"
            else:
                return f"{int(round(millions))} million"
        elif num >= 1000:         # Thousands
            if num <= 10_000:
                return f"{num/1000:.1f}".rstrip('0').rstrip('.') + "k"
            else:
                return f"{round(num/1000)}k"
        return str(num)
    def replacer(match):
        num = float(match.group(0).replace(",", ""))
        return humanize(num)
    if isinstance(text, int | float):
        return humanize(text)
    return re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", replacer, text)