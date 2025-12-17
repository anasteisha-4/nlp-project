import re
import unicodedata
from typing import List, Optional


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = "".join(
        char for char in text 
        if unicodedata.category(char) != "Cc" or char in "\n\t"
    )
    text = text.replace("\t", " ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()
    
    return text


def normalize_text(text: str, lowercase: bool = True) -> str:
    text = clean_text(text)

    if lowercase:
        text = text.lower()

    return text


def prepare_text(
    text: str, 
    lowercase: bool = False,
    max_length: Optional[int] = None
) -> str:
    text = clean_text(text)
    if lowercase:
        text = text.lower()
    if max_length and len(text) > max_length:
        text = text[:max_length]
    return text


def prepare_batch(
    texts: List[str],
    lowercase: bool = False,
    max_length: Optional[int] = None
) -> List[str]:
    return [prepare_text(t, lowercase, max_length) for t in texts]


if __name__ == "__main__":
    test_cases = [
        "  Hello   World!  ",
        "Zażółć gęślą\t\tjaźń",
        "Multiple\n\n\nNewlines",
        "Unicode: привет мир 你好世界",
    ]

    print("Preprocessing tests:")
    for text in test_cases:
        cleaned = prepare_text(text)
        print(f"  '{text}' -> '{cleaned}'")
