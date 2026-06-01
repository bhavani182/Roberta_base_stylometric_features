import json
import re
import random
from copy import deepcopy
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"C:\\Users\\saval\\OneDrive\\Desktop\\Thesis\\Roberta_base_stylometric_features\\data\\val.jsonl"
OUTPUT_FILE_AUGMENTED = r"C:\Users\saval\OneDrive\Desktop\Thesis\Roberta_base_stylometric_features\obfuscated_data\val_pan_obfuscated_augmented.jsonl"
OUTPUT_FILE_ONLY_OBF = r"C:\Users\saval\OneDrive\Desktop\Thesis\Roberta_base_stylometric_features\obfuscated_data\val_pan_obfuscated_only.jsonl"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Probability controls for stochastic transformations
P_APPLY_SYNONYM = 0.25
P_APPLY_SENTENCE_SPLIT = 0.20
P_APPLY_SENTENCE_MERGE = 0.15
P_APPLY_FILLER_REMOVAL = 0.30
P_APPLY_FUNCTION_WORD_BALANCE = 0.30

# Whether to save original + obfuscated together
SAVE_AUGMENTED = True

# Whether to save only obfuscated data
SAVE_ONLY_OBFUSCATED = True


# ============================================================
# SIMPLE RESOURCES
# ============================================================

# Lightweight contraction expansion
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'m": " am",
    "'ve": " have",
    "'ll": " will",
    "'d": " would"
}

# Small safe synonym map
# Keep this conservative so meaning is not damaged too much.
SYNONYM_MAP = {
    "big": ["large", "major"],
    "small": ["little", "minor"],
    "important": ["significant", "notable"],
    "show": ["demonstrate", "display"],
    "shows": ["demonstrates", "displays"],
    "use": ["utilize"],
    "uses": ["utilizes"],
    "help": ["assist", "support"],
    "helps": ["assists", "supports"],
    "start": ["begin"],
    "starts": ["begins"],
    "end": ["finish", "conclude"],
    "ends": ["finishes", "concludes"],
    "buy": ["purchase"],
    "bought": ["purchased"],
    "get": ["obtain", "receive"],
    "got": ["obtained", "received"],
    "good": ["fine", "solid"],
    "bad": ["poor", "weak"],
    "many": ["numerous", "many"],
    "more": ["additional", "more"],
    "less": ["reduced", "less"],
    "easy": ["simple"],
    "hard": ["difficult"],
    "idea": ["concept", "notion"],
    "ideas": ["concepts", "notions"],
    "result": ["outcome"],
    "results": ["outcomes", "findings"],
    "method": ["approach"],
    "methods": ["approaches"],
    "problem": ["issue"],
    "problems": ["issues"],
    "different": ["distinct", "different"],
    "same": ["identical", "same"],
    "make": ["create", "produce"],
    "made": ["created", "produced"]
}

# Common fillers / discourse markers that often carry style
FILLERS = {
    "really", "very", "actually", "basically", "literally",
    "just", "quite", "rather", "maybe", "perhaps"
}

# Small function-word pool for balancing stylistic extremes
FUNCTION_WORDS = ["however", "therefore", "moreover", "thus", "indeed"]

# Simple stopwords for safe function-word checks
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "for", "to", "from", "and", "but", "or", "if", "then", "that", "this"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_sentences(text):
    """
    Light sentence splitter.
    Keeps punctuation boundaries simple and robust.
    """
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def join_sentences(sentences):
    text = " ".join(s.strip() for s in sentences if s.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text):
    return re.findall(r"\b\w+\b|\S", text, flags=re.UNICODE)


# ============================================================
# MASKING / NORMALIZATION
# ============================================================

def mask_urls_emails_usernames(text):
    # emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)

    # urls
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # social usernames like @name
    text = re.sub(r'(?<!\w)@\w+', '[USER]', text)

    return text


def mask_numbers(text):
    # replace digits but preserve rough structure
    return re.sub(r'\d', '0', text)


def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_punctuation(text):
    # collapse repeated punctuation
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)

    # normalize space before punctuation
    text = re.sub(r'\s+([!?.,;:])', r'\1', text)

    # ensure single space after punctuation where appropriate
    text = re.sub(r'([.!?;:,])([^\s"\')\]])', r'\1 \2', text)

    return normalize_whitespace(text)


def lowercase_normalize(text):
    return text.lower()


def expand_contractions(text):
    for k, v in CONTRACTIONS.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    return text


def remove_fillers(text):
    def repl(match):
        word = match.group(0)
        return "" if word.lower() in FILLERS else word
    text = re.sub(r'\b\w+\b', repl, text)
    text = normalize_whitespace(text)
    return text


def simple_named_placeholder_mask(text):
    

    # quoted usernames / handles already handled separately

    # probable long IDs
    text = re.sub(r'\b[A-Z]{2,}\d{2,}[A-Z0-9]*\b', '[ID]', text)

    # probable dates
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)

    # basic capitalized full names: John Smith
    text = re.sub(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', '[NAME]', text)

    return text


# ============================================================
# STYLOMETRIC PERTURBATION
# ============================================================

def synonym_substitution(text, prob=P_APPLY_SYNONYM):
    """
    Conservative synonym replacement.
    Applies to a subset of words only.
    """
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text, flags=re.UNICODE)
    out = []

    for tok in tokens:
        low = tok.lower()
        if re.fullmatch(r'\w+', tok) and low in SYNONYM_MAP and random.random() < prob:
            replacement = random.choice(SYNONYM_MAP[low])

            # Preserve capitalization pattern roughly
            if tok.istitle():
                replacement = replacement.title()
            elif tok.isupper():
                replacement = replacement.upper()

            out.append(replacement)
        else:
            out.append(tok)

    text = ""
    for i, tok in enumerate(out):
        if i == 0:
            text += tok
        else:
            if re.match(r'[^\w\s]', tok):
                text += tok
            elif re.match(r'[^\w\s]', out[i-1]):
                if out[i-1] in ['"', "(", "[", "{", "/"]:
                    text += tok
                else:
                    text += " " + tok
            else:
                text += " " + tok

    return normalize_whitespace(text)


def normalize_sentence_length(text):
    """
    Try to reduce extreme sentence length patterns:
    - split very long sentences
    - merge very short consecutive sentences
    """
    sentences = split_sentences(text)
    if not sentences:
        return text

    # Split very long sentences
    new_sentences = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 30 and random.random() < P_APPLY_SENTENCE_SPLIT:
            # split near midpoint at comma if possible
            comma_positions = [i for i, w in enumerate(words) if ',' in w]
            midpoint = len(words) // 2

            split_idx = None
            if comma_positions:
                split_idx = min(comma_positions, key=lambda x: abs(x - midpoint))
            else:
                split_idx = midpoint

            left = " ".join(words[:split_idx]).strip(", ")
            right = " ".join(words[split_idx:]).strip(", ")

            if left:
                if not re.search(r'[.!?]$', left):
                    left += "."
                new_sentences.append(left)
            if right:
                if not re.search(r'[.!?]$', right):
                    right += "."
                new_sentences.append(right)
        else:
            new_sentences.append(sent)

    # Merge very short sentences
    merged = []
    i = 0
    while i < len(new_sentences):
        current = new_sentences[i]
        current_len = len(current.split())

        if (
            i < len(new_sentences) - 1 and
            current_len < 5 and
            random.random() < P_APPLY_SENTENCE_MERGE
        ):
            nxt = new_sentences[i + 1]
            current = re.sub(r'[.!?]+$', '', current) + ", " + nxt[0].lower() + nxt[1:]
            merged.append(current)
            i += 2
        else:
            merged.append(current)
            i += 1

    return join_sentences(merged)


def function_word_balance(text):
    """
    Reduce extreme stylistic spikes by lightly inserting
    or replacing with common discourse markers.
    """
    if random.random() > P_APPLY_FUNCTION_WORD_BALANCE:
        return text

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text

    idx = random.randrange(1, len(sentences))
    marker = random.choice(FUNCTION_WORDS)

    sent = sentences[idx].strip()
    if sent and sent.split()[0].lower() not in FUNCTION_WORDS:
        # prepend marker lightly
        if sent[0].isupper():
            sent = marker.capitalize() + ", " + sent[0].lower() + sent[1:]
        else:
            sent = marker + ", " + sent
        sentences[idx] = sent

    return join_sentences(sentences)


def reduce_repetition(text):
    # repeated words: "very very very"
    text = re.sub(r'\b(\w+)(\s+\1\b){1,}', r'\1', text, flags=re.IGNORECASE)
    return text


def normalize_capitalization(text):
    """
    Mostly lowercase, but preserve placeholders like [EMAIL].
    """
    placeholder_map = {}
    matches = re.findall(r'\[[A-Z_]+\]', text)

    for i, ph in enumerate(set(matches)):
        key = f"__PH_{i}__"
        placeholder_map[key] = ph
        text = text.replace(ph, key)

    text = text.lower()

    for key, ph in placeholder_map.items():
        text = text.replace(key.lower(), ph)

    # capitalize first character if possible
    if text:
        text = text[0].upper() + text[1:]

    return text


# ============================================================
# MASTER OBFUSCATION PIPELINE
# ============================================================

def pan_style_obfuscate(text):
    """
    PAN-style practical obfuscation:
    preserve content, reduce style-identifying signals.
    """
    if not isinstance(text, str):
        return text

    text = text.strip()
    if not text:
        return text

    # 1) protect obvious metadata-like artifacts
    text = mask_urls_emails_usernames(text)
    text = simple_named_placeholder_mask(text)

    # 2) normalize easy stylistic cues
    text = expand_contractions(text)
    text = mask_numbers(text)
    text = reduce_repetition(text)
    text = normalize_punctuation(text)

    # 3) remove some fillers
    if random.random() < P_APPLY_FILLER_REMOVAL:
        text = remove_fillers(text)

    # 4) lexical perturbation
    text = synonym_substitution(text)

    # 5) sentence-level stylometric normalization
    text = normalize_sentence_length(text)
    text = function_word_balance(text)

    # 6) capitalization normalization
    text = normalize_capitalization(text)

    # 7) final cleanup
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    return text


# ============================================================
# DATASET BUILDERS
# ============================================================

def build_augmented_dataset(records):
    """
    For each original sample:
    - keep original
    - add obfuscated version with same label
    """
    augmented = []

    for rec in records:
        if "text" not in rec:
            augmented.append(rec)
            continue

        original = deepcopy(rec)
        augmented.append(original)

        obf = deepcopy(rec)
        obf["text"] = pan_style_obfuscate(obf["text"])

        # add metadata to track source
        obf["obfuscated"] = True
        obf["obfuscation_type"] = "pan_style"

        # optionally modify id so duplicates do not collide
        if "id" in obf:
            obf["id"] = f"{obf['id']}_obf"

        augmented.append(obf)

    return augmented


def build_only_obfuscated_dataset(records):
    obf_records = []

    for rec in records:
        new_rec = deepcopy(rec)
        if "text" in new_rec:
            new_rec["text"] = pan_style_obfuscate(new_rec["text"])
            new_rec["obfuscated"] = True
            new_rec["obfuscation_type"] = "pan_style"
        if "id" in new_rec:
            new_rec["id"] = f"{new_rec['id']}_obf"
        obf_records.append(new_rec)

    return obf_records


# ============================================================
# MAIN
# ============================================================

def main():
    records = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(records)} records from {INPUT_FILE}")

    # preview
    print("\n--- SAMPLE PREVIEW ---")
    for rec in records[:3]:
        text = rec.get("text", "")
        obf = pan_style_obfuscate(text)
        print("\nORIGINAL:")
        print(text[:500])
        print("\nOBFUSCATED:")
        print(obf[:500])
        print("-" * 80)

    if SAVE_AUGMENTED:
        augmented = build_augmented_dataset(records)
        save_jsonl(augmented, OUTPUT_FILE_AUGMENTED)
        print(f"Saved augmented dataset to: {OUTPUT_FILE_AUGMENTED}")
        print(f"Total records in augmented dataset: {len(augmented)}")

    if SAVE_ONLY_OBFUSCATED:
        only_obf = build_only_obfuscated_dataset(records)
        save_jsonl(only_obf, OUTPUT_FILE_ONLY_OBF)
        print(f"Saved obfuscated-only dataset to: {OUTPUT_FILE_ONLY_OBF}")
        print(f"Total records in obfuscated-only dataset: {len(only_obf)}")


if __name__ == "__main__":
    main()