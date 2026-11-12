"""Unit tests for the social media text cleaning and signal extraction pipeline."""

from __future__ import annotations

from src.data_pipeline.cleaner import SocialTextCleaner, clean_social_text


def test_emoji_translation_and_hashtag_unpacking() -> None:
    cleaner = SocialTextCleaner(use_ekphrasis=False)
    result = cleaner.clean("You are a 🤡 #IdentityHate #StopCyberBullying")

    assert "clown face" in result.cleaned_text
    assert "identity hate" in result.cleaned_text
    assert "stop cyber bullying" in result.cleaned_text
    assert result.hashtags == ["IdentityHate", "StopCyberBullying"]
    assert result.emojis == ["🤡"]


def test_obfuscated_slur_restoration() -> None:
    cleaner = SocialTextCleaner(use_ekphrasis=False)
    result = cleaner.clean("Shut up you @ss $lut b!tch")

    assert "ass" in result.cleaned_text
    assert "slut" in result.cleaned_text
    assert "bitch" in result.cleaned_text
    assert result.obfuscation_hits == 3


def test_urls_mentions_and_contractions() -> None:
    cleaner = SocialTextCleaner(use_ekphrasis=False)
    result = cleaner.clean("@alice don't click https://example.com/spam ok?")

    assert "<user>" in result.cleaned_text
    assert "<url>" in result.cleaned_text
    assert "do not" in result.cleaned_text
    assert result.url_count == 1
    assert result.mention_count == 1


def test_elongation_and_sentiment_signal() -> None:
    cleaner = SocialTextCleaner(use_ekphrasis=False)
    positive = cleaner.clean("Love this awesome community!!!")
    negative = cleaner.clean("I hate this stupid trash game")

    assert positive.sentiment_score > 0
    assert positive.exclamation_count == 3
    assert negative.sentiment_score < 0
    assert clean_social_text("Whyyyyy though") == "whyy though"
