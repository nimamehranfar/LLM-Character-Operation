from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import string
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import SUPPORTED_OPERATIONS
from src.executor.operations import CharacterExecutor, CharacterOperation, ExecutionRequest

TOTAL = 50_000
SPLIT_COUNTS = {"train": 35_000, "dev": 5_000, "test": 5_000, "heldout": 5_000}
NEGATIVE_FRACTION = 0.37
POSITIVE_FRACTION = 0.63
SOURCE_FRACTIONS = {"real_word": 0.30, "augmented_word": 0.50, "random_generated": 0.20}
CWUM_STYLE_FRACTION = 0.30  # layered on top; native project style remains the majority
SEED = 20260919
PRINTABLE = string.ascii_letters + string.digits + "!#$%&()*+,-./:;<=>?@[]^_{}~"

# Sentence-scoped operations corresponding to CWUM English task families.
SENTENCE_OPS = {
    "WORD_COUNT", "WORD_LENGTH_AT", "INSERT_TEXT_IN_WORD", "INSERT_WORDS",
    "CHAR_AT_IN_WORD", "REVERSE_WORD_AT",
}

TRAIN_PREFIX = (
    "", "Please compute this exactly: ", "Literal-text task: ", "Character-level request: ",
    "Use the text exactly as supplied. ", "Exact string operation: ", "Quick exact check: ",
    "Preserve case and symbols. ", "Do not normalize the input. ",
)
TRAIN_SUFFIX = (
    "", " Return only the exact result.", " Preserve the literal input.", " No approximation.",
    " Give only the answer.", " Treat positions as one-based where stated.",
)
HELDOUT_PREFIX = (
    "Audit this sequence without rewriting it first: ",
    "I need a mechanically exact outcome for the following request: ",
    "Interpret the quoted material literally and perform just this operation: ",
    "Resolve this orthographic request with exact indexing: ",
    "For this unseen phrasing, compute only the requested transformation: ",
)
HELDOUT_SUFFIX = (
    " Respond with the result alone.",
    " Do not substitute a nearby operation.",
    " Maintain every unmentioned symbol unchanged.",
    " Use the stated indexing convention exactly.",
)


def load_words(path: Path) -> list[str]:
    words = [x.strip().lower() for x in path.read_text(encoding="utf-8").splitlines()]
    words = sorted({w for w in words if w.isascii() and w.isalpha() and 4 <= len(w) <= 16})
    if len(words) < 500:
        raise RuntimeError(f"Need >=500 real words, found {len(words)}")
    return words


def q(text: str, rng: random.Random) -> str:
    left, right = rng.choice((("\"", "\""), ("'", "'"), ("`", "`")))
    return f"{left}{text}{right}"


def random_token(rng: random.Random, min_len: int = 4, max_len: int = 28) -> str:
    n = rng.randint(min_len, max_len)
    mode = rng.choices(("alpha", "alnum", "punct", "repeat"), weights=(30, 25, 25, 20), k=1)[0]
    if mode == "alpha":
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))
    if mode == "alnum":
        return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))
    if mode == "punct":
        return "".join(rng.choice(PRINTABLE) for _ in range(n))
    a, b, c = rng.sample(string.ascii_lowercase, 3)
    return "".join(rng.choices((a, b, c), weights=(7, 2, 1), k=n))


def augment_word(word: str, rng: random.Random) -> str:
    chars = list(word)
    for _ in range(rng.randint(2, 7)):
        mode = rng.choice(("repeat", "insert", "duplicate", "substitute", "case", "punct"))
        if mode == "repeat" and chars:
            i = rng.randrange(len(chars)); chars[i:i+1] = [chars[i]] * rng.randint(2, 7)
        elif mode == "insert":
            chars.insert(rng.randrange(len(chars)+1), rng.choice(string.ascii_lowercase + string.digits))
        elif mode == "duplicate" and len(chars) >= 2:
            a = rng.randrange(len(chars)-1); b = min(len(chars), a + rng.randint(1, 5)); chars[b:b] = chars[a:b]
        elif mode == "substitute" and chars:
            chars[rng.randrange(len(chars))] = rng.choice(string.ascii_lowercase)
        elif mode == "case" and chars:
            i = rng.randrange(len(chars)); chars[i] = chars[i].upper()
        elif mode == "punct":
            chars.insert(rng.randrange(len(chars)+1), rng.choice("_-.:@+!?"))
        if len(chars) >= 96:
            break
    return "".join(chars[:96])


def one_token(source: str, words: list[str], rng: random.Random) -> str:
    if source == "real_word":
        return rng.choice(words)
    if source == "augmented_word":
        return augment_word(rng.choice(words), rng)
    return random_token(rng)


def sentence(source: str, words: list[str], rng: random.Random) -> str:
    n = rng.randint(3, 10)
    if source == "real_word":
        return " ".join(rng.choice(words) for _ in range(n))
    if source == "augmented_word":
        toks = [rng.choice(words) for _ in range(n)]
        mutate = rng.sample(range(n), k=rng.randint(1, max(1, min(n, 4))))
        for i in mutate:
            toks[i] = augment_word(toks[i], rng)
        return " ".join(toks)
    return " ".join(random_token(rng, 3, 14) for _ in range(rng.randint(3, 7)))


def choose_char(text: str, rng: random.Random) -> str:
    if rng.random() < 0.18:
        absent = [c for c in string.ascii_letters + string.digits + "_-!?" if c not in text]
        if absent:
            return rng.choice(absent)
    return rng.choice(text)


def insertion_text(source: str, words: list[str], rng: random.Random, *, words_only: bool = False) -> str:
    if words_only:
        return " ".join(rng.choice(words) for _ in range(rng.randint(1, 3)))
    if rng.random() < 0.55:
        return rng.choice(string.ascii_letters + string.digits + "_-!?@")
    return one_token(source, words, rng)[: rng.randint(1, 5)]


def make_args(op: str, source: str, words: list[str], rng: random.Random) -> dict[str, Any]:
    text = sentence(source, words, rng) if op in SENTENCE_OPS else one_token(source, words, rng)
    args: dict[str, Any] = {"text": text}
    if op in {"COUNT_CHAR", "FIND_CHAR"}:
        args["character"] = choose_char(text, rng)
    if op == "CHAR_AT":
        args["index"] = rng.randint(1, len(text))
    elif op == "INSERT_TEXT":
        args["insertion"] = insertion_text(source, words, rng)
        args["index"] = rng.randint(1, len(text) + 1)
    elif op in {"WORD_LENGTH_AT", "REVERSE_WORD_AT"}:
        args["word_index"] = rng.randint(1, len(text.split()))
    elif op == "CHAR_AT_IN_WORD":
        wi = rng.randint(1, len(text.split())); word = text.split()[wi - 1]
        args["word_index"] = wi; args["index"] = rng.randint(1, len(word))
    elif op == "INSERT_TEXT_IN_WORD":
        wi = rng.randint(1, len(text.split())); word = text.split()[wi - 1]
        args["insertion"] = insertion_text(source, words, rng)
        args["word_index"] = wi; args["index"] = rng.randint(1, len(word) + 1)
    elif op == "INSERT_WORDS":
        args["insertion"] = insertion_text("real_word", words, rng, words_only=True)
        args["word_index"] = rng.randint(1, len(text.split()) + 1)
    return args


NATIVE_TRAIN = {
    "COUNT_CHAR": [
        "How many times does {character} occur in {text}?", "Count exact matches of {character} inside {text}.",
        "Give the frequency of literal {character} in {text}.", "In {text}, what is the number of positions equal to {character}?",
        "Scan {text} and tally only {character}.", "Return count(text={text}, target={character}) using exact case.",
    ],
    "STRING_LENGTH": [
        "How many characters are in {text}?", "Give the exact character length of {text}.",
        "Count every position in {text} once.", "What is len({text}) at the literal-character level?",
        "Measure {text} from first character through last.", "Return the total symbol count for {text}.",
    ],
    "FIND_CHAR": [
        "Find the first one-based position of {character} in {text}; use 0 if absent.",
        "Where is the earliest {character} in {text} when indexing starts at 1? Return 0 if missing.",
        "Locate the first exact {character} within {text}; one-based, absent=0.",
        "Search {text} left-to-right for {character} and give the first position or zero.",
        "Return first_index_1based(text={text}, target={character}), with 0 for no match.",
    ],
    "WORD_COUNT": [
        "How many whitespace-separated words are in {text}?", "Count the words in {text} using whitespace boundaries.",
        "Give the word total for {text}.", "How many tokens separated by spaces appear in {text}?",
        "Return the number of words in this sentence: {text}.",
    ],
    "WORD_LENGTH_AT": [
        "How many characters are in word {word_index} of {text}?", "Give the length of the {word_index}th word in {text}.",
        "Count characters in word number {word_index} from {text}.", "For {text}, measure the word at one-based index {word_index}.",
    ],
    "INSERT_TEXT": [
        "Insert {insertion} into {text} before one-based position {index}.", "Place {insertion} at character boundary {index} in {text}.",
        "Modify {text} by inserting {insertion} just before position {index}.", "At one-based insertion point {index}, add {insertion} to {text}.",
    ],
    "INSERT_TEXT_IN_WORD": [
        "In word {word_index} of {text}, insert {insertion} before character position {index}.",
        "Add {insertion} at position {index} inside the {word_index}th word of {text}.",
        "Edit only word {word_index} in {text}: insert {insertion} at one-based boundary {index}.",
    ],
    "INSERT_WORDS": [
        "Insert the words {insertion} before word boundary {word_index} in {text}.",
        "Add {insertion} at one-based word insertion point {word_index} of {text}.",
        "Place {insertion} into {text} immediately before word position {word_index}.",
    ],
    "CHAR_AT": [
        "What character is at one-based position {index} in {text}?", "Return character number {index} from {text}.",
        "Select the symbol at index {index} of {text}, counting from 1.", "Give {text}[{index}] under one-based indexing.",
    ],
    "CHAR_AT_IN_WORD": [
        "What is character {index} of word {word_index} in {text}?", "From word {word_index} of {text}, return its {index}th character.",
        "Select one-based character {index} inside one-based word {word_index} of {text}.",
    ],
    "REVERSE": [
        "Reverse the characters of {text}.", "Write {text} in exact reverse character order.",
        "Return the characterwise reversal of {text}.", "Flip the sequence {text} end-to-start.",
    ],
    "REVERSE_WORD_AT": [
        "Reverse only word {word_index} in {text}.", "In {text}, flip the characters of the {word_index}th word and keep all other words unchanged.",
        "Character-reverse word number {word_index} of {text}.",
    ],
}

# CWUM-style structures are deliberately included without the external benchmark's "Code prohibited" phrase.
CWUM_TRAIN = {
    "STRING_LENGTH": ["Count the letters in the word {text}.", "How many letters does the word {text} contain?"],
    "WORD_COUNT": ["Count the words in the sentence {text}.", "How many words does this sentence contain: {text}?"],
    "WORD_LENGTH_AT": ["Count the letters in word {word_index} of the sentence {text}.", "How many letters are in the {word_index}th word of {text}?"],
    "INSERT_TEXT": ["Insert {insertion} into the word {text} at position {index}.", "Add {insertion} at position {index} of {text}."],
    "INSERT_TEXT_IN_WORD": ["Insert {insertion} at position {index} in word {word_index} of the sentence {text}."],
    "INSERT_WORDS": ["Insert {insertion} at word position {word_index} in the sentence {text}."],
    "CHAR_AT": ["Identify letter {index} in the word {text}.", "What is the {index}th letter of {text}?"],
    "CHAR_AT_IN_WORD": ["Identify letter {index} of word {word_index} in the sentence {text}."],
    "REVERSE": ["Reverse the word {text}.", "Write the word {text} backwards."],
    "REVERSE_WORD_AT": ["Reverse word {word_index} in the sentence {text}."],
    "COUNT_CHAR": ["Count the occurrences of {character} in {text}."],
    "FIND_CHAR": ["Find the first position of {character} in {text}; use 0 when absent."],
}

HELDOUT_TEMPLATES = {
    "COUNT_CHAR": ["For sequence {text}, what multiplicity belongs to symbol {character}?", "Tally positions occupied by {character} across {text}, with case preserved."],
    "STRING_LENGTH": ["If every displayed symbol of {text} gets one slot, how many slots are occupied?", "What is the cardinality of the character sequence {text}?"],
    "FIND_CHAR": ["Number {text} from 1 onward; which number belongs to the earliest {character}, or 0 if none?", "Return min one-based coordinate matching {character} in {text}; empty match set maps to zero."],
    "WORD_COUNT": ["Partition {text} only at whitespace; how many resulting words are there?", "What is the whitespace-delimited word cardinality of {text}?"],
    "WORD_LENGTH_AT": ["Enumerate the words of {text} from one; how many characters occupy entry {word_index}?", "Measure the orthographic width of word slot {word_index} in {text}."],
    "INSERT_TEXT": ["Splice {insertion} into {text} at boundary {index}, where boundary 1 precedes the first character.", "At insertion coordinate {index} of {text}, place literal {insertion} without altering neighbors."],
    "INSERT_TEXT_IN_WORD": ["Within word slot {word_index} of {text}, splice {insertion} at internal boundary {index}.", "Edit word {word_index} only: place {insertion} at its boundary {index} in {text}."],
    "INSERT_WORDS": ["Splice word sequence {insertion} into {text} at word boundary {word_index}.", "Using one-based word boundaries, place {insertion} at coordinate {word_index} in {text}."],
    "CHAR_AT": ["Enumerate characters of {text} from one; which symbol occupies coordinate {index}?", "Read exactly the symbol residing at one-based slot {index} of {text}."],
    "CHAR_AT_IN_WORD": ["In one-based word slot {word_index} of {text}, which symbol occupies internal coordinate {index}?", "Index words then characters from one: return ({word_index},{index}) from {text}."],
    "REVERSE": ["Apply the character-order involution to {text}: first becomes last and so on.", "Mirror the complete character sequence {text} from end to beginning."],
    "REVERSE_WORD_AT": ["In {text}, mirror only the character sequence occupying word slot {word_index}.", "Keep word order fixed but reverse the internal characters of word {word_index} in {text}."],
}


def render_template(template: str, args: dict[str, Any], rng: random.Random) -> str:
    vals = {k: (q(str(v), rng) if k in {"text", "character", "insertion"} else str(v)) for k, v in args.items()}
    return template.format(**vals)


def make_positive(op: str, source: str, split: str, style: str, words: list[str], rng: random.Random, used: set[str], serial: int) -> dict[str, Any]:
    executor = CharacterExecutor()
    for _ in range(5000):
        args = make_args(op, source, words, rng)
        if split == "heldout":
            template = rng.choice(HELDOUT_TEMPLATES[op])
            prompt = rng.choice(HELDOUT_PREFIX) + render_template(template, args, rng) + rng.choice(HELDOUT_SUFFIX)
            family = "heldout_exclusive"
            generation_style = "heldout_native"
        else:
            if style == "cwum_style":
                template = rng.choice(CWUM_TRAIN[op]); generation_style = "cwum_style_reconstruction"; family = "cwum_train"
            else:
                template = rng.choice(NATIVE_TRAIN[op]); generation_style = "native"; family = "native_train"
            prompt = rng.choice(TRAIN_PREFIX) + render_template(template, args, rng) + rng.choice(TRAIN_SUFFIX)
        prompt = prompt.strip()
        if prompt in used:
            continue
        request = ExecutionRequest(
            operation=CharacterOperation(op), text=str(args["text"]),
            character=None if "character" not in args else str(args["character"]),
            index=None if "index" not in args else int(args["index"]),
            word_index=None if "word_index" not in args else int(args["word_index"]),
            insertion=None if "insertion" not in args else str(args["insertion"]),
        )
        expected = str(executor.execute(request))
        used.add(prompt)
        return {
            "example_id": f"final_{split}_task_{serial:05d}", "split": split, "operation": op,
            "category": source, "prompt": prompt, "expected": expected, "arguments": args,
            "length_regime": "long" if len(str(args["text"])) > 48 else "iid",
            "generation_style": generation_style, "source_style": source, "template_family": family,
        }
    raise RuntimeError("could not create unique positive prompt")


HARD_NEGATIVE_CATEGORIES = (
    "vowel_count", "consonant_count", "digit_count", "uppercase_count", "lowercase_count",
    "distinct_char_count", "substring_count", "contains_char", "last_occurrence", "nth_occurrence",
    "remove_char", "replace_char", "sort_chars", "palindrome", "case_convert", "byte_length",
    "token_count", "line_count", "count_before_first", "count_after_first", "case_insensitive_count",
    "first_word", "last_word", "word_at", "delete_word", "swap_chars", "rotate_string",
)


def hard_negative(text: str, split: str, rng: random.Random) -> tuple[str, str]:
    cat = rng.choice(HARD_NEGATIVE_CATEGORIES)
    T = q(text, rng); ch = q(rng.choice(text), rng); ch2 = q(rng.choice(string.ascii_letters), rng)
    n = rng.randint(2, 5)
    if split == "heldout":
        bodies = {
            "vowel_count": f"Determine the cardinality of the vowel subset inside {T}.",
            "consonant_count": f"How many alphabetic consonant positions does {T} contain?",
            "digit_count": f"Report the number of positions in {T} whose symbols belong to 0 through 9.",
            "uppercase_count": f"How many cased letters of {T} are uppercase?", "lowercase_count": f"How many cased letters of {T} are lowercase?",
            "distinct_char_count": f"What is the size of the set of unique symbols occurring in {T}?",
            "substring_count": f"Count occurrences of the multi-character fragment {q(text[:min(3,len(text))],rng)} in {T}.",
            "contains_char": f"Return a boolean indicating whether {T} contains {ch}.",
            "last_occurrence": f"Give the greatest one-based coordinate occupied by {ch} in {T}, or zero.",
            "nth_occurrence": f"Locate occurrence number {n} of {ch} in {T}, or zero when it does not exist.",
            "remove_char": f"Delete every {ch} from {T} and return the transformed sequence.",
            "replace_char": f"Substitute {ch2} for every {ch} throughout {T}.", "sort_chars": f"Return the symbols of {T} in lexical order.",
            "palindrome": f"Decide whether {T} reads identically in both directions.", "case_convert": f"Map every alphabetic symbol of {T} to uppercase.",
            "byte_length": f"How many UTF-8 bytes encode {T}?", "token_count": f"How many tokenizer tokens would {T} occupy?",
            "line_count": f"How many text lines are represented by {T}?", "count_before_first": f"How many symbols precede the earliest {ch} in {T}?",
            "count_after_first": f"How many symbols follow the earliest {ch} in {T}?", "case_insensitive_count": f"Count {ch} in {T} after ignoring case.",
            "first_word": f"Return the first whitespace-delimited word from {T}.", "last_word": f"Return the final whitespace-delimited word from {T}.",
            "word_at": f"Return word number {n} from {T}.", "delete_word": f"Delete word number {n} from {T}.",
            "swap_chars": f"Exchange the first and last characters of {T}.", "rotate_string": f"Rotate {T} left by {n} character positions.",
        }
    else:
        bodies = {
            "vowel_count": f"How many vowels are in {T}?", "consonant_count": f"Count consonants in {T}.", "digit_count": f"Count numeric digits in {T}.",
            "uppercase_count": f"Count uppercase letters in {T}.", "lowercase_count": f"Count lowercase letters in {T}.", "distinct_char_count": f"How many different characters occur in {T}?",
            "substring_count": f"How often does substring {q(text[:min(3,len(text))],rng)} appear in {T}?", "contains_char": f"Does {T} contain {ch}? Answer yes or no.",
            "last_occurrence": f"Find the last one-based position of {ch} in {T}.", "nth_occurrence": f"Find the {n}th occurrence position of {ch} in {T}.",
            "remove_char": f"Remove all {ch} characters from {T}.", "replace_char": f"Replace every {ch} in {T} with {ch2}.", "sort_chars": f"Sort characters of {T}.",
            "palindrome": f"Is {T} a palindrome?", "case_convert": f"Convert {T} to uppercase.", "byte_length": f"What is the UTF-8 byte length of {T}?",
            "token_count": f"How many model tokens are in {T}?", "line_count": f"Count lines in {T}.", "count_before_first": f"How many characters come before the first {ch} in {T}?",
            "count_after_first": f"How many characters come after the first {ch} in {T}?", "case_insensitive_count": f"Count {ch} in {T} ignoring case.",
            "first_word": f"What is the first word of {T}?", "last_word": f"What is the last word of {T}?", "word_at": f"What is word {n} of {T}?",
            "delete_word": f"Remove word {n} from {T}.", "swap_chars": f"Swap the first and last characters of {T}.", "rotate_string": f"Rotate {T} left by {n} positions.",
        }
    return cat, bodies[cat]


GENERAL_TRAIN = (
    ("factual", "Explain {topic} in two concise sentences."), ("translation", "Translate '{phrase}' into Italian."),
    ("summarization", "Summarize this idea in one bullet: {topic} affects everyday decisions."),
    ("planning", "Plan a short museum visit in {place}."), ("creative", "Write a four-line story about a lighthouse at sunrise."),
    ("math", "Compute {a} * {b} + {c}."), ("coding", "Write a Python function that merges two sorted lists."),
    ("advice", "Give three practical tips for organizing a study schedule."), ("classification", "Classify '{thing}' as animal, plant, mineral, or other."),
)
GENERAL_HELDOUT = (
    ("factual", "Give a beginner-level account of {topic}, without discussing spelling."),
    ("translation", "Render the phrase '{phrase}' naturally in Spanish."),
    ("summarization", "Compress the following notion to one sentence: {topic} influences policy and science."),
    ("planning", "Design a compact walking itinerary for {place}."),
    ("creative", "Compose a tiny fictional scene involving a train in winter."),
    ("math", "Evaluate the arithmetic expression ({a}+{b})*{c}."),
    ("coding", "Describe an algorithm for breadth-first graph traversal."),
    ("advice", "Suggest two ways to keep research notes organized."),
    ("classification", "Assign '{thing}' to a broad semantic category."),
)


def general_negative(split: str, rng: random.Random) -> tuple[str, str]:
    bank = GENERAL_HELDOUT if split == "heldout" else GENERAL_TRAIN
    cat, template = rng.choice(bank)
    prompt = template.format(
        topic=rng.choice(("photosynthesis", "inflation", "volcanoes", "databases", "gravity", "ocean currents")),
        phrase=rng.choice(("good morning", "see you tomorrow", "thank you for your help")), place=rng.choice(("Rome", "Oslo", "Kyoto", "Lisbon")),
        a=rng.randint(2,50), b=rng.randint(2,50), c=rng.randint(2,20), thing=rng.choice(("oak", "salmon", "granite", "violin", "copper")),
    )
    return cat, prompt


def make_negative(split: str, words: list[str], rng: random.Random, used: set[str], serial: int) -> dict[str, Any]:
    for _ in range(5000):
        if rng.random() < 0.70:
            # Use both single-token and sentence material to create semantic-neighbor negatives.
            raw = sentence(rng.choice(tuple(SOURCE_FRACTIONS)), words, rng) if rng.random() < 0.35 else one_token(rng.choice(tuple(SOURCE_FRACTIONS)), words, rng)
            category, prompt = hard_negative(raw, split, rng)
            family = "heldout_hard_negative" if split == "heldout" else "hard_negative"
        else:
            category, prompt = general_negative(split, rng)
            family = "heldout_general_negative" if split == "heldout" else "general_negative"
        if split == "heldout":
            prompt = rng.choice(HELDOUT_PREFIX) + prompt + rng.choice(HELDOUT_SUFFIX)
        else:
            prompt = rng.choice(TRAIN_PREFIX) + prompt + rng.choice(TRAIN_SUFFIX)
        prompt = prompt.strip()
        if prompt in used:
            continue
        used.add(prompt)
        return {
            "example_id": f"final_{split}_control_{serial:05d}", "split": split, "operation": "NONE",
            "category": category, "prompt": prompt, "expected": "", "arguments": {}, "length_regime": "control",
            "generation_style": "heldout_negative" if split == "heldout" else "native_negative",
            "source_style": "control", "template_family": family,
        }
    raise RuntimeError("could not create unique negative prompt")


def exact_quota(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {k: total * v for k, v in fractions.items()}
    out = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(out.values())
    for key in sorted(raw, key=lambda k: (raw[k] - int(raw[k]), k), reverse=True)[:remainder]:
        out[key] += 1
    return out


def operation_quota(total: int, offset: int = 0) -> dict[str, int]:
    ops = list(SUPPORTED_OPERATIONS)
    base, rem = divmod(total, len(ops))
    out = {op: base for op in ops}
    for i in range(rem):
        out[ops[(i + offset) % len(ops)]] += 1
    return out


def build_split(split: str, total: int, words: list[str], rng: random.Random, used: set[str], op_offset: int) -> list[dict[str, Any]]:
    negative_n = int(round(total * NEGATIVE_FRACTION))
    positive_n = total - negative_n
    source_quota = exact_quota(positive_n, SOURCE_FRACTIONS)
    op_quota = operation_quota(positive_n, op_offset)

    # Build operation and source assignment lists independently, then shuffle to keep both exact marginals.
    op_assign = [op for op, n in op_quota.items() for _ in range(n)]
    source_assign = [src for src, n in source_quota.items() for _ in range(n)]
    rng.shuffle(op_assign); rng.shuffle(source_assign)
    style_assign = ["cwum_style"] * int(round(positive_n * CWUM_STYLE_FRACTION))
    style_assign += ["native"] * (positive_n - len(style_assign))
    rng.shuffle(style_assign)

    rows: list[dict[str, Any]] = []
    for i, (op, source, style) in enumerate(zip(op_assign, source_assign, style_assign)):
        if split == "heldout":
            style = "heldout"
        rows.append(make_positive(op, source, split, style, words, rng, used, i))
    for i in range(negative_n):
        rows.append(make_negative(split, words, rng, used, i))
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest_for(all_rows: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    flat = [r for rows in all_rows.values() for r in rows]
    prompts = [r["prompt"] for r in flat]
    train_families = {r["template_family"] for r in all_rows["train"]}
    held_families = {r["template_family"] for r in all_rows["heldout"]}
    return {
        "dataset": "character_operations_50k",
        "seed": seed,
        "total": len(flat),
        "split_counts": {k: len(v) for k, v in all_rows.items()},
        "negative_count": sum(r["operation"] == "NONE" for r in flat),
        "negative_fraction": sum(r["operation"] == "NONE" for r in flat) / len(flat),
        "positive_count": sum(r["operation"] != "NONE" for r in flat),
        "positive_source_counts": dict(Counter(r["source_style"] for r in flat if r["operation"] != "NONE")),
        "operation_counts": dict(Counter(r["operation"] for r in flat if r["operation"] != "NONE")),
        "generation_style_counts": dict(Counter(r["generation_style"] for r in flat)),
        "supported_operations": list(SUPPORTED_OPERATIONS),
        "native_positive_source_mix": SOURCE_FRACTIONS,
        "cwum_style_positive_fraction_nonheldout": CWUM_STYLE_FRACTION,
        "cwum_code_prohibited_in_primary_dataset": False,
        "heldout_template_families_disjoint_from_train": train_families.isdisjoint(held_families),
        "exact_prompt_duplicates": len(prompts) - len(set(prompts)),
        "sha256_prompts": hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/character_operations_50k")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--words", default="data/resources/faker_en_words.txt")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    words = load_words(REPO_ROOT / args.words)
    used: set[str] = set()
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for offset, (split, total) in enumerate(SPLIT_COUNTS.items()):
        all_rows[split] = build_split(split, total, words, rng, used, op_offset=offset * 3)

    out_dir = REPO_ROOT / args.output_dir
    filenames = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl", "heldout": "heldout.jsonl"}
    for split, filename in filenames.items():
        write_jsonl(out_dir / filename, all_rows[split])
    manifest = manifest_for(all_rows, args.seed)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
