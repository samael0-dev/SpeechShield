"""Social media text normalization, emoji translation, and slang/obfuscation cleaning."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import emoji

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_]+)")
CAMEL_SPLIT_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")

CONTRACTIONS: dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "u": "you",
    "ur": "your",
    "idk": "i do not know",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "stfu": "shut the fuck up",
    "gtfo": "get the fuck out",
    "kys": "kill yourself",
    "af": "as fuck",
}

OBFUSCATED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\w)\$lut(s)?(?!\w)", re.IGNORECASE), r"slut\1"),
    (re.compile(r"(?<!\w)@ss(hole|es)?(?!\w)", re.IGNORECASE), r"ass\1"),
    (re.compile(r"(?<!\w)a\$\$(hole|es)?(?!\w)", re.IGNORECASE), r"ass\1"),
    (re.compile(r"(?<!\w)b[!1*]tch(es|ing)?(?!\w)", re.IGNORECASE), r"bitch\1"),
    (re.compile(r"(?<!\w)f[*u@#]+ck(er|ers|ing|ed|s)?(?!\w)", re.IGNORECASE), r"fuck\1"),
    (re.compile(r"(?<!\w)f[\W_]*u[\W_]*c[\W_]*k(?!\w)", re.IGNORECASE), "fuck"),
    (re.compile(r"(?<!\w)sh[!1*]t(ty|s|ting)?(?!\w)", re.IGNORECASE), r"shit\1"),
    (re.compile(r"(?<!\w)d[!1*]ck(head|s)?(?!\w)", re.IGNORECASE), r"dick\1"),
    (re.compile(r"(?<!\w)c[*u]nt(s)?(?!\w)", re.IGNORECASE), r"cunt\1"),
    (re.compile(r"(?<!\w)wh[0o]re(s)?(?!\w)", re.IGNORECASE), r"whore\1"),
    (re.compile(r"(?<!\w)r[e3]tard(ed|s)?(?!\w)", re.IGNORECASE), r"retard\1"),
    (re.compile(r"(?<!\w)f[@a]gg?[0o]t(s)?(?!\w)", re.IGNORECASE), r"faggot\1"),
    (re.compile(r"(?<!\w)n[!1]gg?[e3a@](r|rs|s)?(?!\w)", re.IGNORECASE), r"nigger\1"),
    (re.compile(r"(?<!\w)k[!1]ll\s+y[0o]urs[e3]lf(?!\w)", re.IGNORECASE), "kill yourself"),
]

POSITIVE_LEXICON = {
    "love", "great", "awesome", "nice", "good", "happy", "thanks", "thank",
    "helpful", "kind", "amazing", "cool", "respect", "support", "appreciate",
    "welcome", "gg", "proud", "excellent", "wonderful",
}
NEGATIVE_LEXICON = {
    "hate", "stupid", "idiot", "trash", "loser", "garbage", "pathetic", "ugly",
    "kill", "die", "shut", "worst", "disgusting", "scum", "moron", "dumb",
    "toxic", "clown", "fuck", "shit", "bitch", "ass", "slut", "whore", "retard",
}


@dataclass(slots=True)
class CleanedTextResult:
    """Structured output from the social media text cleaning pipeline."""

    raw_text: str
    cleaned_text: str
    hashtags: list[str] = field(default_factory=list)
    emojis: list[str] = field(default_factory=list)
    obfuscation_hits: int = 0
    url_count: int = 0
    mention_count: int = 0
    uppercase_ratio: float = 0.0
    exclamation_count: int = 0
    sentiment_score: float = 0.0
    used_ekphrasis: bool = False


class SocialTextCleaner:
    """Cleans noisy social media text while preserving toxicity cues.

    Supports optional `ekphrasis` normalization with a deterministic built-in
    pipeline for hashtag unpacking, emoji translation, contraction expansion,
    and obfuscated slur restoration.
    """

    def __init__(
        self,
        use_ekphrasis: bool = True,
        lowercase: bool = True,
        replace_urls: bool = True,
        replace_mentions: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.replace_urls = replace_urls
        self.replace_mentions = replace_mentions
        self._ekphrasis_processor: Any = None
        if use_ekphrasis:
            self._ekphrasis_processor = self._init_ekphrasis()

    @staticmethod
    def _init_ekphrasis() -> Any:
        try:
            from ekphrasis.classes.preprocessor import TextPreProcessor
            from ekphrasis.classes.tokenizer import SocialTokenizer
            from ekphrasis.dicts.emoticons import emoticons

            return TextPreProcessor(
                normalize=["url", "email", "percent", "money", "phone", "user", "time", "date", "number"],
                annotate={"hashtag"},
                all_caps_tag="wrap",
                fix_html=True,
                segmenter="twitter",
                corrector="twitter",
                unpack_hashtags=True,
                unpack_contractions=True,
                spell_correct_elong=False,
                tokenizer=SocialTokenizer(lowercase=False).tokenize,
                dicts=[emoticons],
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - optional dependency
            logger.debug("ekphrasis unavailable, using built-in social cleaner: %s", exc)
            return None

    @staticmethod
    def unpack_hashtag(match: re.Match[str]) -> str:
        """Split camel-cased or underscored hashtags into plain words."""
        tag = match.group(1)
        spaced = tag.replace("_", " ")
        spaced = CAMEL_SPLIT_PATTERN.sub(" ", spaced)
        return spaced.strip()

    @staticmethod
    def translate_emojis(text: str) -> tuple[str, list[str]]:
        """Convert Unicode emojis into readable text descriptions."""
        found = [item["emoji"] for item in emoji.emoji_list(text)]
        if not found:
            return text, []
        demojized = emoji.demojize(text, delimiters=(" ", " "))
        # Turn ":clown_face:" style tokens into "clown face"
        demojized = re.sub(r"\b([a-z0-9]+)_([a-z0-9_]+)\b", lambda m: m.group(0).replace("_", " "), demojized)
        return demojized, found

    @staticmethod
    def deobfuscate_slurs(text: str) -> tuple[str, int]:
        """Normalize common symbol/leetspeak obfuscated profanity and slurs."""
        hits = 0
        updated = text
        for pattern, replacement in OBFUSCATED_PATTERNS:
            updated, count = pattern.subn(replacement, updated)
            hits += count
        return updated, hits

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand common English contractions and social shorthand."""
        tokens = text.split()
        expanded: list[str] = []
        for token in tokens:
            stripped = token.strip(".,!?;:\"'()")
            lower = stripped.lower()
            if lower in CONTRACTIONS:
                prefix = token[: len(token) - len(token.lstrip(".,!?;:\"'()"))]
                suffix = token[len(stripped) + len(prefix) :]
                expanded.append(f"{prefix}{CONTRACTIONS[lower]}{suffix}")
            else:
                expanded.append(token)
        return " ".join(expanded)

    @staticmethod
    def compute_sentiment_score(cleaned_text: str) -> float:
        """Compute lightweight lexicon polarity score in [-1.0, 1.0]."""
        words = re.findall(r"[a-z']+", cleaned_text.lower())
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in POSITIVE_LEXICON)
        neg = sum(1 for w in words if w in NEGATIVE_LEXICON)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 4)

    def clean(self, text: str) -> CleanedTextResult:
        """Run full social media cleaning pipeline and return structured signals."""
        raw_text = text or ""
        alpha_chars = [c for c in raw_text if c.isalpha()]
        uppercase_ratio = (
            round(sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars), 4)
            if alpha_chars
            else 0.0
        )
        exclamation_count = raw_text.count("!")

        hashtags = HASHTAG_PATTERN.findall(raw_text)
        url_matches = URL_PATTERN.findall(raw_text)

        working = raw_text
        used_ekphrasis = False

        # Translate emojis early before punctuation stripping
        working, emojis = self.translate_emojis(working)

        if self._ekphrasis_processor is not None:  # pragma: no cover - optional
            try:
                tokens = self._ekphrasis_processor.pre_process_doc(working)
                working = " ".join(tokens)
                working = re.sub(r"</?hashtag>|</?allcaps>", " ", working)
                used_ekphrasis = True
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("ekphrasis failed on input, falling back: %s", exc)

        if self.replace_urls:
            working = URL_PATTERN.sub(" <url> ", working)
        else:
            working = URL_PATTERN.sub(" ", working)

        # Restore obfuscated slurs before @mention replacement so "@ss" is not treated as "@user"
        working, obfuscation_hits = self.deobfuscate_slurs(working)

        mention_matches = MENTION_PATTERN.findall(working)
        if self.replace_mentions:
            working = MENTION_PATTERN.sub(" <user> ", working)
        else:
            working = MENTION_PATTERN.sub(" ", working)

        # Unpack hashtags after URLs (avoids splitting URL fragments)
        working = HASHTAG_PATTERN.sub(self.unpack_hashtag, working)

        # Collapse excessive elongations (e.g., "whyyyyy" -> "whyy")
        working = ELONGATION_PATTERN.sub(r"\1\1", working)

        # Expand contractions & internet shorthand
        working = self.expand_contractions(working)

        if self.lowercase:
            working = working.lower()

        working = WHITESPACE_PATTERN.sub(" ", working).strip()
        sentiment_score = self.compute_sentiment_score(working)

        return CleanedTextResult(
            raw_text=raw_text,
            cleaned_text=working,
            hashtags=hashtags,
            emojis=emojis,
            obfuscation_hits=obfuscation_hits,
            url_count=len(url_matches),
            mention_count=len(mention_matches),
            uppercase_ratio=uppercase_ratio,
            exclamation_count=exclamation_count,
            sentiment_score=sentiment_score,
            used_ekphrasis=used_ekphrasis,
        )


_DEFAULT_CLEANER = SocialTextCleaner(use_ekphrasis=False)


def clean_social_text(text: str) -> str:
    """Convenience function returning cleaned social media text."""
    return _DEFAULT_CLEANER.clean(text).cleaned_text
