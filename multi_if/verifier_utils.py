"""Ground-truth-aligned verifier utility functions and registry.

Provides standardized text processing utilities that match IFBench/IFEval
ground truth implementations. Used in two ways:
  1. At generation time: rendered into prompts so the LLM uses them
  2. At runtime: imported by generated verifier code via
     `from verifier_utils import count_words, ...`

Source of truth: ifeval-suite/if_bench/instructions_util.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import string
from typing import Any

# Optional runtime dependencies (declared in ifeval-suite/if_bench/requirements.txt).
# Imported lazily so the module loads even when these are missing; functions
# that require them raise on call.
try:
    import emoji as _emoji
except ImportError:  # pragma: no cover
    _emoji = None

try:
    import syllapy as _syllapy
except ImportError:  # pragma: no cover
    _syllapy = None

# ---------------------------------------------------------------------------
# Ground-truth utility implementations
# ---------------------------------------------------------------------------

# --- Sentence splitter constants (from instructions_util.py:1564-1571) ---
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|"
    r"Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a rule-based approach.

    Handles abbreviations (Dr., Mr., U.S.A.), decimal numbers, ellipses,
    websites, and quote-punctuation interactions. Matches the IFBench/IFEval
    ground truth implementation exactly.

    Source: ifeval-suite/if_bench/instructions_util.py:1574-1622
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>",
        text,
    )
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "\u201c" in text:
        text = text.replace(".\u201d", "\u201d.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text: str) -> int:
    r"""Count words using \w+ tokenization.

    Equivalent to nltk.tokenize.RegexpTokenizer(r"\w+") but without the
    nltk dependency. Contractions and hyphens are split into separate tokens.

    Source: ifeval-suite/if_bench/instructions_util.py:1625-1630
    """
    return len(re.findall(r"\w+", text))


def count_sentences(text: str) -> int:
    """Count sentences using the rule-based splitter.

    Uses split_into_sentences() (not nltk punkt) for consistency with
    IFBench eval. The ground truth is actually inconsistent here (IFBench
    training uses punkt for counting but rule-based for splitting), but we
    standardize on rule-based for correctness on abbreviations.
    """
    return len(split_into_sentences(text))


def count_paragraphs(text: str) -> int:
    r"""Count paragraphs by splitting on double newlines.

    Matches the IFBench ParagraphChecker pattern: split on '\n\n' and
    filter empty results.
    """
    return len([p for p in text.split("\n\n") if p.strip()])


def count_keyword(text: str, keyword: str) -> int:
    r"""Count substring occurrences of keyword, case-insensitive.

    No word boundary — 'theater' contains 'the' (returns 1). This matches
    IFBench KeywordsMultipleChecker / KeywordFrequencyChecker semantics.

    Source: ifeval-suite/if_bench/instructions.py:1877 (KeywordsMultipleChecker)
            ifeval-suite/multi_if/ifeval.py:2528 (KeywordFrequencyChecker)
    """
    if not keyword:
        return 0
    return len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))


def count_keyword_exact(text: str, keyword: str) -> int:
    r"""Count whole-word occurrences of keyword, case-insensitive.

    Uses \b word boundaries — 'theater' does NOT contain 'the' as a word
    (returns 0). This matches IFBench ForbiddenWords semantics.

    Source: ifeval-suite/multi_if/ifeval.py:2880 (ForbiddenWords)
    """
    if not keyword:
        return 0
    return len(
        re.findall(r"\b" + re.escape(keyword) + r"\b", text, flags=re.IGNORECASE)
    )


def is_substring_present(text: str, substring: str) -> bool:
    """Check if substring is present in text, case-insensitive.

    Used for phrase/quote/sentence matching. Matches IFBench
    IncludeKeywordChecker and verbatim-paragraph-include patterns.
    """
    if not substring:
        return True  # vacuous: empty substring is trivially present
    return substring.lower() in text.lower()


# ---------------------------------------------------------------------------
# Markdown-structure utilities
# ---------------------------------------------------------------------------

_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_SETEXT_H1_RE = re.compile(r"^(.+)\n=+[ \t]*$", re.MULTILINE)
_SETEXT_H2_RE = re.compile(r"^(.+)\n-+[ \t]*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.DOTALL)
_INDENT_CODE_RE = re.compile(r"(?:^|\n)((?:    |\t).*(?:\n(?:    |\t).*)*)", re.MULTILINE)
_BLOCKQUOTE_LINE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?(.*)$", re.MULTILINE)
_HTML_BLOCKQUOTE_RE = re.compile(r"<blockquote[^>]*>(.*?)</blockquote>", re.DOTALL | re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*|__([^_\n]+?)__")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)|(?<!_)_([^_\n]+?)_(?!_)")


def extract_headings(text: str) -> list[str]:
    """Extract markdown heading TEXT (stripped of `#`/`=`/`-`).

    Supports ATX (`# heading`) and Setext (`heading\\n====`) styles.
    Returns list of heading strings in order of appearance.
    """
    headings: list[str] = []
    for m in _ATX_HEADING_RE.finditer(text):
        headings.append(m.group(2).strip())
    for m in _SETEXT_H1_RE.finditer(text):
        headings.append(m.group(1).strip())
    for m in _SETEXT_H2_RE.finditer(text):
        headings.append(m.group(1).strip())
    return headings


def extract_code_blocks(text: str) -> list[str]:
    """Extract contents of fenced (```...```) and indented code blocks.

    Returns list of code-block contents (without the fence/indent markers).
    """
    blocks: list[str] = []
    # Fenced code blocks — consume first so they don't trigger indent matcher
    remaining = text
    fenced: list[tuple[int, int, str]] = []
    for m in _CODE_FENCE_RE.finditer(text):
        fenced.append((m.start(), m.end(), m.group(1)))
        blocks.append(m.group(1))
    # Remove fenced regions for indent detection
    if fenced:
        parts = []
        prev = 0
        for s, e, _ in fenced:
            parts.append(text[prev:s])
            prev = e
        parts.append(text[prev:])
        remaining = "\n".join(parts)
    # Indented code blocks (4 spaces or tab)
    for m in _INDENT_CODE_RE.finditer(remaining):
        content = m.group(1)
        # Strip the leading 4 spaces / tab from each line
        dedented = "\n".join(
            (line[4:] if line.startswith("    ") else (line[1:] if line.startswith("\t") else line))
            for line in content.split("\n")
        )
        blocks.append(dedented.strip())
    return [b for b in blocks if b.strip()]


def extract_blockquotes(text: str) -> list[str]:
    """Extract contents of markdown `>` blockquotes and HTML `<blockquote>` tags.

    Returns list of blockquote contents (lines joined with spaces within a quote).
    """
    quotes: list[str] = []
    # HTML blockquotes
    for m in _HTML_BLOCKQUOTE_RE.finditer(text):
        inner = re.sub(r"<[^>]+>", "", m.group(1))
        quotes.append(inner.strip())
    # Markdown blockquotes (contiguous `> ` lines)
    current: list[str] = []
    for line in text.split("\n"):
        m = _BLOCKQUOTE_LINE_RE.match(line)
        if m is not None:
            current.append(m.group(1))
        else:
            if current:
                quotes.append(" ".join(current).strip())
                current = []
    if current:
        quotes.append(" ".join(current).strip())
    return [q for q in quotes if q]


def extract_bold(text: str) -> list[str]:
    r"""Extract markdown bold contents (`**text**` or `__text__`).

    Returns list of bold text strings in order of appearance.
    """
    out: list[str] = []
    for m in _BOLD_RE.finditer(text):
        out.append((m.group(1) or m.group(2)).strip())
    return out


def extract_italic(text: str) -> list[str]:
    r"""Extract markdown italic contents (`*text*` or `_text_`).

    Note: does not match `**` (bold) — uses negative lookaround.
    """
    out: list[str] = []
    for m in _ITALIC_RE.finditer(text):
        out.append((m.group(1) or m.group(2)).strip())
    return out


# ---------------------------------------------------------------------------
# Format validators (IFBench-aligned)
# ---------------------------------------------------------------------------


def validate_json(text: str) -> bool:
    """Check if text is a valid JSON response.

    Strips optional markdown code fence (```json / ```Json / ```JSON / ```)
    before validating via json.loads(). Matches IFBench JsonFormat exactly.

    Source: ifeval-suite/multi_if/ifeval.py:2681-2713 (JsonFormat)
    """
    stripped = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def validate_csv(text: str, delimiter: str = ",") -> bool:
    """Check if text parses as valid CSV with consistent column counts.

    Uses csv.reader on io.StringIO. Returns True iff text parses without
    exception, has ≥1 row, and every row has the same column count as the
    first (header) row.

    Extracted as the primitive shared by IFBench CityCSVChecker,
    SpecialCharacterCSVChecker, and QuotesCSVChecker. Those checkers add
    schema-specific header/column assertions on top of this primitive.

    Source: ifeval-suite/if_bench/instructions.py:1726-1836 (CSV checkers)
    """
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)
    except Exception:
        return False
    if not rows:
        return False
    num_cols = len(rows[0])
    return all(len(row) == num_cols for row in rows)


# ---------------------------------------------------------------------------
# Phonetic / orthographic utilities (IFBench-aligned)
# ---------------------------------------------------------------------------


def count_syllables(word: str) -> int:
    """Count syllables in a single word using syllapy (IFBench's choice).

    Requires the `syllapy` package (declared in ifeval-suite's requirements).

    Source: ifeval-suite/if_bench/instructions.py:1122-1142
            (AlternateParitySyllablesChecker calls syllapy.count(word))
    """
    if _syllapy is None:
        raise ImportError("syllapy is required for count_syllables")
    return _syllapy.count(word)


def is_emoji(char: str) -> bool:
    """Check if a single character is an emoji using the emoji library.

    Requires the `emoji` package (declared in ifeval-suite's requirements).

    Source: ifeval-suite/if_bench/instructions.py:846-882
            (EmojiSentenceChecker calls emoji.is_emoji(char))
    """
    if _emoji is None:
        raise ImportError("emoji is required for is_emoji")
    return _emoji.is_emoji(char)


def count_emojis(text: str) -> int:
    """Count emoji characters in text (iterates characters via emoji.is_emoji).

    Convenience wrapper over is_emoji for whole-text counts.
    """
    if _emoji is None:
        raise ImportError("emoji is required for count_emojis")
    return sum(1 for ch in text if _emoji.is_emoji(ch))


def is_palindrome(s: str) -> bool:
    """Check if s is a palindrome after canonicalization.

    Canonicalization (matches IFBench PalindromeChecker): strip punctuation,
    strip whitespace, lowercase. A non-empty canonicalized string is a
    palindrome iff it equals its reverse.

    IFBench's PalindromeChecker iterates single words (via value.lower().split()
    after punctuation strip); this utility generalizes to arbitrary strings so
    multi-word palindromes like "was it a car or a cat I saw" also work.

    Source: ifeval-suite/if_bench/instructions.py:604-625 (PalindromeChecker)
    """
    normalized = s.translate(str.maketrans("", "", string.punctuation))
    normalized = "".join(normalized.split()).lower()
    return len(normalized) > 0 and normalized == normalized[::-1]


# ---------------------------------------------------------------------------
# Utility registry: behavioral specs + examples for prompt rendering
# ---------------------------------------------------------------------------

UTILS_REGISTRY: dict[str, dict[str, Any]] = {
    "count_words": {
        "signature": "count_words(text: str) -> int",
        "spec": r"Count words using \w+ tokenization (word characters only). "
                "Contractions and hyphens are split into separate tokens.",
        "examples": [
            ('count_words("It\'s well-known")', 4, "contractions/hyphens split into separate tokens"),
            ('count_words("Hello, world!")', 2, "punctuation is stripped"),
            ('count_words("Dr. Smith\'s office")', 4, "possessive 's is a separate token"),
            ('count_words("")', 0, "empty string"),
        ],
        "source": "ifbench:if_bench/instructions_util.py:1625-1630 (count_words); "
                  "uses re.findall(r'\\w+') instead of nltk.RegexpTokenizer(r'\\w+') "
                  "to avoid the nltk dependency (functionally equivalent).",
    },
    "count_sentences": {
        "signature": "count_sentences(text: str) -> int",
        "spec": "Count sentences using rule-based splitting that handles "
                "abbreviations (Dr., Mr., Mrs., U.S.A., Ph.D.), decimal numbers, "
                "and ellipses as non-sentence-boundaries.",
        "examples": [
            ('count_sentences("Dr. Smith went home. He rested.")', 2,
             "abbreviation 'Dr.' is not a sentence boundary"),
            ('count_sentences("The U.S.A. is big. Really big.")', 2,
             "acronym 'U.S.A.' is not a sentence boundary"),
            ('count_sentences("Hello world.")', 1, "single sentence"),
            ('count_sentences("Wait... really? Yes!")', 3,
             "ellipsis, question mark, and exclamation are boundaries"),
        ],
        "source": "derived: wraps split_into_sentences (IFBench). "
                  "IFEval's count uses nltk.punkt; we prefer rule-based for "
                  "abbreviation correctness.",
    },
    "count_paragraphs": {
        "signature": "count_paragraphs(text: str) -> int",
        "spec": r"Count paragraphs by splitting on double newlines (\n\n). "
                "Empty paragraphs (only whitespace) are not counted.",
        "examples": [
            (r'count_paragraphs("Para one.\n\nPara two.")', 2, "standard double newline"),
            (r'count_paragraphs("Single paragraph.")', 1, "no double newline"),
            (r'count_paragraphs("A.\n\nB.\n\nC.")', 3, "three paragraphs"),
        ],
        "source": "ifbench: matches ParagraphChecker \\n\\n-split pattern",
    },
    "split_into_sentences": {
        "signature": "split_into_sentences(text: str) -> list[str]",
        "spec": "Split text into a list of sentence strings using rule-based "
                "splitting. Handles abbreviations, acronyms, decimals, ellipses, "
                "and quote-punctuation interactions.",
        "examples": [
            ('split_into_sentences("Dr. Smith went home. He rested.")',
             '["Dr. Smith went home.", "He rested."]', "abbreviation preserved"),
            ('split_into_sentences("Hello! How are you? Fine.")',
             '["Hello!", "How are you?", "Fine."]', "multiple terminators"),
        ],
        "source": "ifbench:if_bench/instructions_util.py:1574-1622 (split_into_sentences) — "
                  "verbatim port",
    },
    "count_keyword": {
        "signature": "count_keyword(text: str, keyword: str) -> int",
        "spec": "Count SUBSTRING occurrences of keyword (case-insensitive). "
                "'theater' contains 'the' once. Use for 'contain X N times', "
                "'include phrase X', 'contain the phrase X'.",
        "examples": [
            ('count_keyword("the theater", "the")', 2, "substring; 'the' matches both occurrences"),
            ('count_keyword("theater", "the")', 1, "substring match, NOT word boundary"),
            ('count_keyword("The Theater", "the")', 2, "case-insensitive"),
            ('count_keyword("state-of-the-art", "the")', 1, "substring within hyphens"),
        ],
        "source": "ifeval:multi_if/ifeval.py:2589-2595 (KeywordFrequencyChecker); "
                  "adds re.escape(keyword) for regex-metachar safety (IFBench omits it).",
    },
    "count_keyword_exact": {
        "signature": "count_keyword_exact(text: str, keyword: str) -> int",
        "spec": r"Count WHOLE-WORD occurrences of keyword using \b word "
                "boundaries (case-insensitive). 'theater' does NOT contain "
                "'the' as a word. Use for 'the word X' semantics, especially "
                "forbidden-word constraints ('do not use the word X').",
        "examples": [
            ('count_keyword_exact("the theater", "the")', 1, "'the' matches, 'theater' does not"),
            ('count_keyword_exact("theater", "the")', 0, "not a whole word — no match"),
            ('count_keyword_exact("The Theater", "the")', 1, "case-insensitive whole word"),
            ('count_keyword_exact("the-the", "the")', 2, "hyphen is a word boundary"),
            ('count_keyword_exact("don\'t stop", "don")', 1, "apostrophe is a word boundary so 'don' matches in \"don't\""),
        ],
        "source": "ifeval:multi_if/ifeval.py:2913-2918 (ForbiddenWords); "
                  "adds re.escape(keyword) + .findall (IFBench uses re.search without escape).",
    },
    "is_substring_present": {
        "signature": "is_substring_present(text: str, substring: str) -> bool",
        "spec": "Check if substring is present in text (case-insensitive). "
                "Use for 'include the quote X', 'include verbatim', phrase matching.",
        "examples": [
            ('is_substring_present("Hello world", "ello")', True, "substring present"),
            ('is_substring_present("Hello world", "goodbye")', False, "not present"),
            ('is_substring_present("The quick brown fox", "QUICK")', True, "case-insensitive"),
        ],
        "source": "ifbench:if_bench/instructions.py:1072-1077 (IncludeKeywordChecker); "
                  "same lowercase-substring primitive, generalized from per-sentence to whole-text.",
    },
    "extract_headings": {
        "signature": "extract_headings(text: str) -> list[str]",
        "spec": "Extract markdown heading text (stripped of `#`/`=`/`-` markers). "
                "Supports ATX (`# title`) and Setext (`title\\n====`) styles.",
        "examples": [
            ('extract_headings("# Title\\n\\nBody")', '["Title"]', "ATX heading"),
            ('extract_headings("## A\\n\\n### B")', '["A", "B"]', "multiple headings"),
            ('extract_headings("Heading\\n======")', '["Heading"]', "Setext H1"),
            ('extract_headings("plain text no heading")', '[]', "no headings"),
        ],
        "source": "manual",
    },
    "extract_code_blocks": {
        "signature": "extract_code_blocks(text: str) -> list[str]",
        "spec": "Extract contents of fenced (```) and 4-space-indented code blocks. "
                "Fence language tags (e.g. ```python) are stripped.",
        "examples": [
            ('extract_code_blocks("```python\\nx=1\\n```")', '["x=1"]', "fenced python"),
            ('extract_code_blocks("```\\nhi\\n```\\nprose\\n```\\nbye\\n```")', '["hi", "bye"]', "two fenced"),
            ('extract_code_blocks("no code here")', '[]', "no code"),
        ],
        "source": "manual",
    },
    "extract_blockquotes": {
        "signature": "extract_blockquotes(text: str) -> list[str]",
        "spec": "Extract contents of markdown `>` blockquotes AND HTML `<blockquote>` tags. "
                "Contiguous `>` lines are joined into one quote.",
        "examples": [
            ('extract_blockquotes("> hello\\n> world")', '["hello world"]', "contiguous joined"),
            ('extract_blockquotes("<blockquote>quote</blockquote>")', '["quote"]', "HTML tag"),
            ('extract_blockquotes("plain text")', '[]', "no quotes"),
        ],
        "source": "manual",
    },
    "extract_bold": {
        "signature": "extract_bold(text: str) -> list[str]",
        "spec": "Extract markdown bold contents (`**text**` or `__text__`). "
                "Does not match italic (single `*` or `_`).",
        "examples": [
            ('extract_bold("**hi** and __there__")', '["hi", "there"]', "both bold styles"),
            ('extract_bold("*italic* only")', '[]', "italic not matched"),
        ],
        "source": "manual",
    },
    "extract_italic": {
        "signature": "extract_italic(text: str) -> list[str]",
        "spec": "Extract markdown italic contents (`*text*` or `_text_`, single marker). "
                "Does not match bold (`**` or `__`).",
        "examples": [
            ('extract_italic("*hi* and _there_")', '["hi", "there"]', "both italic styles"),
            ('extract_italic("**bold**")', '[]', "bold not matched"),
        ],
        "source": "manual",
    },
    "validate_json": {
        "signature": "validate_json(text: str) -> bool",
        "spec": "Return True iff text is valid JSON. Strips optional markdown "
                "code fence (```json, ```Json, ```JSON, ```) before calling "
                "json.loads(). Matches IFBench JsonFormat exactly.",
        "examples": [
            ('validate_json(\'{"a": 1}\')', True, "bare valid JSON"),
            ('validate_json("```json\\n{\\"a\\": 1}\\n```")', True, "fenced JSON"),
            ('validate_json(\'{a: 1}\')', False, "unquoted keys rejected"),
            ('validate_json(\'{"a":1,}\')', False, "trailing comma rejected"),
            ('validate_json("not json")', False, "garbage"),
        ],
        "source": "ifeval:multi_if/ifeval.py:2681-2713 (JsonFormat) — verbatim port",
    },
    "validate_csv": {
        "signature": "validate_csv(text: str, delimiter: str = ',') -> bool",
        "spec": "Return True iff text parses with csv.reader and every row has "
                "the same column count as row 0 (header consistency). Primitive "
                "shared by IFBench CSV checkers, without their schema-specific "
                "header/column assertions.",
        "examples": [
            ('validate_csv("a,b,c\\n1,2,3\\n4,5,6")', True, "3 cols, 3 rows"),
            ('validate_csv("a,b,c\\n1,2")', False, "row 1 has fewer cols"),
            ('validate_csv("a\\tb\\n1\\t2", delimiter=\'\\t\')', True, "tab-delimited"),
            ('validate_csv("")', False, "empty"),
        ],
        "source": "ifbench:if_bench/instructions.py:1726-1836 (CSV checker primitive)",
    },
    "count_syllables": {
        "signature": "count_syllables(word: str) -> int",
        "spec": "Count syllables in a single word using syllapy.count() — "
                "IFBench's library choice. Raises ImportError if syllapy "
                "is not installed.",
        "examples": [
            ('count_syllables("hello")', 2, "standard"),
            ('count_syllables("chocolate")', 3, "3 syllables per CMUdict/syllapy"),
            ('count_syllables("a")', 1, "single vowel word"),
        ],
        "source": "ifbench:if_bench/instructions.py:1122-1142 "
                  "(AlternateParitySyllablesChecker uses syllapy.count)",
    },
    "is_emoji": {
        "signature": "is_emoji(char: str) -> bool",
        "spec": "Check if a single character is an emoji using emoji.is_emoji() — "
                "IFBench's library choice. Unicode-range regex is NOT a valid "
                "substitute (misses U+231A, U+1F900+ ranges).",
        "examples": [
            ('is_emoji("\\U0001F600")', True, "😀 grinning face"),
            ('is_emoji("\\u231A")', True, "⌚ watch (missed by unicode-range regex)"),
            ('is_emoji("a")', False, "plain letter"),
        ],
        "source": "ifbench:if_bench/instructions.py:846-882 "
                  "(EmojiSentenceChecker uses emoji.is_emoji)",
    },
    "count_emojis": {
        "signature": "count_emojis(text: str) -> int",
        "spec": "Count emoji characters in text by iterating and applying "
                "emoji.is_emoji. Use for 'include N emojis' / 'no emojis'.",
        "examples": [
            ('count_emojis("\\U0001F600 hello \\U0001F601")', 2, "two emojis"),
            ('count_emojis("no emoji here")', 0, "none"),
        ],
        "source": "ifbench:derived from EmojiSentenceChecker (same library)",
    },
    "is_palindrome": {
        "signature": "is_palindrome(s: str) -> bool",
        "spec": "Return True iff s (with punctuation and whitespace stripped, "
                "lowercased) equals its reverse. Empty canonical string → False. "
                "Supports both single-word and multi-word palindromes.",
        "examples": [
            ('is_palindrome("racecar")', True, "simple single word"),
            ('is_palindrome("A man, a plan, a canal: Panama")', True,
             "multi-word with punctuation"),
            ('is_palindrome("hello")', False, "not a palindrome"),
            ('is_palindrome("")', False, "empty → False"),
        ],
        "source": "ifbench:if_bench/instructions.py:604-625 (PalindromeChecker); "
                  "generalized from word-level (IFBench) to arbitrary strings.",
    },
}


# ---------------------------------------------------------------------------
# Per-type utility mapping (DEPRECATED — use CONCEPTS-based retrieval)
# ---------------------------------------------------------------------------
#
# `UTILS_BY_TYPE` is retained for backward compatibility only. New callers
# should use `detect_concepts(constraint_text)` and retrieve per-concept
# hints from `CONCEPTS`. Utilities are concepts, and a utility may apply to
# many types — type is the wrong primary key. Scheduled for removal once
# `render_concept_hints` replaces `render_utils_for_prompt` in prompt_v2.py.

UTILS_BY_TYPE: dict[str, list[str]] = {
    "length_count": ["count_words", "count_sentences", "count_paragraphs", "split_into_sentences"],
    "inclusion_exclusion_content": [
        "count_keyword",
        "count_keyword_exact",
        "is_substring_present",
        "split_into_sentences",
        "count_words",
        # Markdown structure (for 'in heading', 'in code block', 'in blockquote', 'in bold/italic')
        "extract_headings",
        "extract_code_blocks",
        "extract_blockquotes",
        "extract_bold",
        "extract_italic",
    ],
    "file_format": [],                   # TODO: add validate_json, validate_csv
    "linguistic_format": [],             # TODO: add POS-tagging utils
    "layout": [
        "extract_headings",
        "extract_bold",
        "extract_italic",
    ],
    "prosody_phonetics_format": [],      # TODO: add count_syllables
    "inclusion_exclusion_types": [],     # TODO: add is_emoji
    "consistency": [],
    "conversion": [],
}


# ---------------------------------------------------------------------------
# Concept registry (primary structure for prompt assembly)
# ---------------------------------------------------------------------------
#
# A "concept" is a primitive semantic unit that appears across ≥1 constraint
# types (word counting, JSON validation, POS tagging, rhyme, etc.). Each
# concept either exposes a utility function (when IFBench/IFEval establishes
# the canonical semantics) or provides text guidance (pointing at a library
# or standard). Concepts are retrieved per-constraint via keyword matching
# so shared primitives are taught once and used consistently across types.
#
# `edge_hint` is a placeholder for Step 7 of the plan — it will carry a
# compact string describing kinds of edges to probe in test cases. Left as
# None for now; do NOT inject into prompts until Step 7 is finalized.

CONCEPTS: dict[str, dict[str, Any]] = {
    # --- IFBench/IFEval-aligned utility concepts ------------------------
    "word_count": {
        "util": "count_words",
        "hint": (
            "Use count_words(text: str) -> int from verifier_utils — "
            "re.findall(r'\\w+') tokenization. Contractions and hyphens split "
            "into separate tokens (\"it's\" = 2, \"state-of-the-art\" = 4). "
            "Import: `from verifier_utils import count_words`. "
            "Never use str.split() or len(text.split()) — diverges on contractions/hyphens."
        ),
        "source": "ifbench:if_bench/instructions_util.py:1625",
        "edge_hint": None,
    },
    "sentence_split": {
        "util": "split_into_sentences",
        "hint": (
            "Use split_into_sentences(text: str) -> list[str] from verifier_utils — "
            "rule-based splitter handling abbreviations (Dr., Mr., U.S.A., Ph.D.), "
            "decimals, ellipses, and quote-punctuation. "
            "Pair with count_sentences(text) for counts. "
            "Import: `from verifier_utils import split_into_sentences, count_sentences`. "
            "Never use re.split on [.!?] or nltk.sent_tokenize — both fail on abbreviations."
        ),
        "source": "ifbench:if_bench/instructions_util.py:1574-1622",
        "edge_hint": None,
    },
    "paragraph_split": {
        "util": "count_paragraphs",
        "hint": (
            "Use count_paragraphs(text: str) -> int from verifier_utils — splits "
            "on double newlines (\\n\\n), ignores whitespace-only paragraphs. "
            "For iteration, split via text.split('\\n\\n') and strip empties. "
            "Import: `from verifier_utils import count_paragraphs`."
        ),
        "source": "ifbench:ParagraphChecker pattern",
        "edge_hint": None,
    },
    "keyword": {
        "util": "count_keyword,count_keyword_exact,is_substring_present",
        "hint": (
            "Three utilities by semantic unit: count_keyword_exact (whole word), "
            "count_keyword (case-insensitive substring), is_substring_present (boolean). "
            "Never use str.count or bare \\b regex — breaks on keywords with punctuation."
        ),
        "source": "ifeval:multi_if/ifeval.py (KeywordFrequencyChecker, ForbiddenWords); "
                  "ifbench:IncludeKeywordChecker",
        "edge_hint": None,
    },
    "markdown_struct": {
        "util": "extract_headings,extract_code_blocks,extract_blockquotes,extract_bold,extract_italic",
        "hint": (
            "Use extract_headings / extract_code_blocks / extract_blockquotes / "
            "extract_bold / extract_italic from verifier_utils. "
            "Never parse markdown by hand — divergent parses across verifiers."
        ),
        "source": "manual — IFBench/IFEval do not cover markdown structure",
        "edge_hint": None,
    },
    "json_valid": {
        "util": "validate_json",
        "hint": (
            "Use validate_json(text: str) -> bool from verifier_utils. "
            "Strips optional ```json/``` fence then calls json.loads(). "
            "Returns True iff valid; rejects trailing commas, single quotes, "
            "and extraneous text. "
            "Import: `from verifier_utils import validate_json`. "
            "Never use regex to validate JSON — regex accepts invalid inputs."
        ),
        "source": "ifeval:multi_if/ifeval.py:2681-2713 (JsonFormat)",
        "edge_hint": None,
    },
    "csv_valid": {
        "util": "validate_csv",
        "hint": (
            "Use validate_csv(text: str, delimiter: str = ',') -> bool from "
            "verifier_utils. Returns True iff csv.reader parses text and all "
            "rows have the same column count as row 0. "
            "Import: `from verifier_utils import validate_csv`. "
            "For non-comma delimiters (tabs, pipes), pass delimiter='\\t' / '|'."
        ),
        "source": "ifbench:if_bench/instructions.py:1726-1836 (CSV checker primitive)",
        "edge_hint": None,
    },
    "syllable": {
        "util": "count_syllables",
        "hint": (
            "Use count_syllables(word: str) -> int from verifier_utils — "
            "wraps syllapy.count(word), IFBench's canonical choice. "
            "Strip punctuation and lowercase the word before calling. "
            "Import: `from verifier_utils import count_syllables`. "
            "Never use vowel-counting regex heuristics — diverge on ~47% of real words."
        ),
        "source": "ifbench:if_bench/instructions.py:1122-1142 "
                  "(AlternateParitySyllablesChecker uses syllapy.count)",
        "edge_hint": None,
    },
    "emoji": {
        "util": "is_emoji,count_emojis",
        "hint": (
            "Use is_emoji(char: str) -> bool or count_emojis(text: str) -> int "
            "from verifier_utils. Both wrap emoji.is_emoji (IFBench's library). "
            "Import: `from verifier_utils import is_emoji, count_emojis`. "
            "Never use unicode-range regex — misses ranges like U+231A (⌚) and U+1F900+ (🤌)."
        ),
        "source": "ifbench:if_bench/instructions.py:846-882 (EmojiSentenceChecker)",
        "edge_hint": None,
    },
    "palindrome": {
        "util": "is_palindrome",
        "hint": (
            "Use is_palindrome(s: str) -> bool from verifier_utils. "
            "Strips punctuation and whitespace, lowercases, then compares to reverse. "
            "Works for both single-word and multi-word palindromes. "
            "Empty input → False. "
            "Import: `from verifier_utils import is_palindrome`."
        ),
        "source": "ifbench:if_bench/instructions.py:604-625 (PalindromeChecker)",
        "edge_hint": None,
    },
    # --- Guidance-only concepts (no util) --------------------------------
    "pos": {
        "util": None,
        "hint": (
            "Use spaCy for POS, verb forms, and clause structure. "
            "token.pos_ for coarse POS; token.tag_ for fine-grained Penn Treebank tags "
            "(VBN=past participle, VBG=gerund/present participle, VB=infinitive); "
            "token.dep_ for clause structure (relcl, advcl, acl) and negation (neg). "
            "Setup: `nlp = spacy.load('en_core_web_sm')`. "
            "IFBench uses spaCy; NLTK RB*/VB* tags acceptable but named differently."
        ),
        "source": "guidance:spacy (aligns with IFBench start_verb)",
        "edge_hint": None,
    },
    "tense": {
        "util": None,
        "hint": (
            "Use spaCy morph: token.morph.get('Tense'). "
            "Combine with token.pos_ to isolate verbs."
        ),
        "source": "guidance:spacy",
        "edge_hint": None,
    },
    "passive_voice": {
        "util": None,
        "hint": (
            "Detect via spaCy dependency parse: a token with token.dep_ == 'auxpass' "
            "signals a passive construction."
        ),
        "source": "guidance:spacy",
        "edge_hint": None,
    },
    "rhyme": {
        "util": None,
        "hint": (
            "Use CMUdict via nltk (`from nltk.corpus import cmudict; d = cmudict.dict()`) "
            "or the `pronouncing` library (`pronouncing.rhymes(word)`). "
            "Two words rhyme iff their final stressed vowel + following phonemes match. "
            "Guard with try/except for out-of-vocabulary words."
        ),
        "source": "guidance:cmudict/pronouncing (no IFBench checker)",
        "edge_hint": None,
    },
    "synonym": {
        "util": None,
        "hint": (
            "Use nltk.corpus.wordnet.synsets(word) and iterate lemma.name(); "
            "lemma.antonyms() for antonyms."
        ),
        "source": "guidance:nltk wordnet (matches existing verifier_prompt hint)",
        "edge_hint": None,
    },
    "url": {
        "util": None,
        "hint": (
            "Use urllib.parse.urlparse; valid iff scheme is http/https/ftp "
            "and netloc is non-empty."
        ),
        "source": "guidance:stdlib (no IFBench checker)",
        "edge_hint": None,
    },
    "date_format": {
        "util": None,
        "hint": (
            "Use datetime.strptime(s, fmt) with explicit format strings "
            "(e.g., YYYYMMDD → '%Y%m%d'). Wrap in try/except ValueError."
        ),
        "source": "guidance:stdlib (no IFBench checker)",
        "edge_hint": None,
    },
    "citation_style": {
        "util": None,
        "hint": (
            "Match APA (Author, Year), Chicago author-date (Author Year), "
            "and IEEE [N] via anchored regex."
        ),
        "source": "guidance:regex (no IFBench checker)",
        "edge_hint": None,
    },
}


CONCEPT_KEYWORDS: dict[str, list[str]] = {
    # --- IFBench/IFEval-aligned utility concepts -------------------------
    "word_count":       [r"\bwords?\b"],
    "sentence_split":   [r"\bsentences?\b"],
    "paragraph_split":  [r"\bparagraphs?\b"],
    "keyword":          [
        r"\bthe word\b", r"\bthe words\b",
        r"\bthe phrase\b", r"\bthe quote\b",
        r"forbidden words?", r"verbatim",
        r"\binclude the following\b",
        r"\bdo not (?:use|mention|include) the\b",
    ],
    "markdown_struct":  [
        r"\bbold\b", r"\bitalic\b",
        r"\bheading(s)?\b", r"\bLevel \d markdown\b",
        r"\bcode block(s)?\b", r"\bblockquote(s)?\b",
        r"\bhighlight(ed)?\b",
        r"\bbullets?\b(?!\s+of)",
    ],
    "json_valid":       [r"\bJSON\b", r"\bjson format\b"],
    "csv_valid":        [r"\bCSV\b"],
    "syllable":         [r"\bsyllables?\b"],
    "emoji":            [r"\bemoji(?:s|es)?\b"],
    "palindrome":       [r"\bpalindromes?\b"],
    # --- Guidance-only concepts -----------------------------------------
    "pos":              [
        # Coarse POS
        r"\badverbs?\b", r"\bverbs?\b", r"\bnouns?\b",
        r"\badjectives?\b", r"\bpronouns?\b",
        r"\bprepositions?\b", r"\bconjunctions?\b",
        r"\bdeterminers?\b", r"\barticles?\b(?!\s+in)",
        r"\binterjections?\b", r"\bmodal\s+verbs?\b",
        # Verb forms (fine-grained)
        r"\bparticiples?\b", r"\bgerunds?\b", r"\binfinitives?\b",
        r"\bauxiliary\b",
        # Clause / dep structure
        r"\brelative\s+clauses?\b", r"\badverbial\s+clauses?\b",
        r"\bsubordinate\s+clauses?\b", r"\bmain\s+clauses?\b",
        r"\bnegations?\b",
    ],
    "tense":            [r"\b(?:past|present|future|present\s+perfect)\s+tense\b"],
    "passive_voice":    [r"\bpassive voice\b", r"\bpassive\b"],
    "rhyme":            [r"\brhymes?\b", r"\brhyming\b", r"\brhyme scheme\b"],
    "synonym":          [r"\bsynonyms?\b", r"\bantonyms?\b"],
    "url":              [r"\bURLs?\b", r"\blinks?\b(?!ed)"],
    "date_format":      [r"YYYY[-/ ]?MM[-/ ]?DD", r"YYYYMMDD", r"\bdate format\b"],
    "citation_style":   [
        r"\bcitation(s)?\b", r"\bAPA\b", r"\bChicago\b",
        r"\bIEEE\b", r"\bauthor[- ]date\b",
    ],
}


def detect_concepts(constraint_text: str) -> list[str]:
    """Return the list of concept names whose keyword patterns match.

    Case-insensitive. Order is stable (insertion order of CONCEPT_KEYWORDS).
    Used at prompt-generation time to assemble concept-specific hints.
    """
    matched: list[str] = []
    for concept, patterns in CONCEPT_KEYWORDS.items():
        if any(re.search(p, constraint_text, re.IGNORECASE) for p in patterns):
            matched.append(concept)
    return matched


def render_concept_hints(concepts: list[str]) -> str:
    """Render the concept-hint section for a generation prompt.

    Returns a formatted string listing hints for each matched concept.
    Utilities and guidance are interleaved; `edge_hint` is NOT rendered
    here — it's reserved for the TEST SUITE section (Step 7).
    """
    if not concepts:
        return ""

    lines: list[str] = ["CONCEPT GUIDANCE (primitives relevant to this constraint):"]
    for concept in concepts:
        entry = CONCEPTS.get(concept)
        if entry is None:
            continue
        lines.append("")
        lines.append(f"  [{concept}]  (source: {entry['source']})")
        # Indent the hint
        for hint_line in entry["hint"].split("\n"):
            lines.append(f"    {hint_line}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_utils_by_names(util_names: list[str]) -> str:
    """Render the utility library section given an explicit list of names.

    Used by concept-based retrieval (detect_concepts → utils). Entries
    without a registry match are skipped silently.
    """
    if not util_names:
        return ""

    lines: list[str] = [
        "VERIFIER UTILITIES (use these — do NOT reimplement):",
        f"Import via: from verifier_utils import {', '.join(util_names)}",
        "",
    ]

    for name in util_names:
        entry = UTILS_REGISTRY.get(name)
        if entry is None:
            continue
        lines.append(f"  {entry['signature']}")
        lines.append(f"    {entry['spec']}")
        lines.append("    Examples:")
        for ex in entry["examples"]:
            call, result, explanation = ex
            lines.append(f"      {call} == {result}  # {explanation}")
        lines.append("")

    return "\n".join(lines)


def render_utils_for_prompt(constraint_type: str) -> str:
    """DEPRECATED — type-indexed renderer kept for backward compat.

    New callers should use `render_utils_by_names` with a list derived
    from `detect_concepts(...)`.
    """
    return render_utils_by_names(UTILS_BY_TYPE.get(constraint_type, []))


def utils_for_concepts(concepts: list[str]) -> list[str]:
    """Derive a de-duplicated ordered list of util names from matched concepts.

    Each CONCEPTS entry's 'util' field may be None, a single name, or a
    comma-separated list. Pulls all non-None entries and preserves order
    of first appearance.
    """
    seen: dict[str, None] = {}
    for c in concepts:
        entry = CONCEPTS.get(c)
        if entry is None or not entry.get("util"):
            continue
        for name in entry["util"].split(","):
            name = name.strip()
            if name and name not in seen:
                seen[name] = None
    return list(seen.keys())
