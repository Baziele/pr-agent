"""Record validation (live-review probe fixture)."""

import re


def is_valid_email(value):
    return re.match(r"^[^@]+@[^@]+\.[^@]+$", value) is not None


def normalize_phone(value):
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    else:
        return None


def classify_score(score):
    if score > 90:
        return "excellent"
    elif score > 70:
        return "good"
    elif score > 50:
        return "fair"
    else:
        return "poor"
