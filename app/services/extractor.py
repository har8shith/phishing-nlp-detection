import re, json, spacy
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
MONEY_PATTERN = re.compile(r"(?:₹|\$|eur|rs\.?)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
_nlp = None
def get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp
ACTION_PATTERNS = [
    ("reset your password", "credential_reset_request"),
    ("verify your account", "account_verification_request"),
    ("confirm your identity", "identity_verification_request"),
]
def extract_information(email_text: str) -> dict:
    urls = URL_PATTERN.findall(email_text)
    emails = EMAIL_PATTERN.findall(email_text)
    money = MONEY_PATTERN.findall(email_text)
    nlp = get_spacy()
    doc = nlp(email_text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    actions = []
    lowered = email_text.lower()
    for phrase, tag in ACTION_PATTERNS:
        if phrase in lowered:
            actions.append(tag)
    return {
        "urls": list(set(urls)),
        "email_addresses": list(set(emails)),
        "money": list(set(money)),
        "organizations": list(set(orgs)),
        "persons": list(set(persons)),
        "dates": list(set(dates)),
        "actions": list(set(actions)),
    }
def serialize_extracted_info(info: dict) -> str:
    return json.dumps(info, ensure_ascii=False)
