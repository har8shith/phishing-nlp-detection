import re
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_PATTERN.sub(" <URL> ", text)
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()
