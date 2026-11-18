"""The original prompt tags stored in a generated image's PNG text chunks.

Images written by SD Runner carry them under the ``SDR_`` names, which are
still read.
"""

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from extensions.image_data_extractor import ImageDataExtractor

POSITIVE = ImageDataExtractor.ORIGINAL_POSITIVE_TAGS_KEY
NEGATIVE = ImageDataExtractor.ORIGINAL_NEGATIVE_TAGS_KEY
(LEGACY_POSITIVE,) = ImageDataExtractor.LEGACY_METADATA_KEYS[POSITIVE]
(LEGACY_NEGATIVE,) = ImageDataExtractor.LEGACY_METADATA_KEYS[NEGATIVE]


def make_png(path, **texts):
    info = PngInfo()
    for key, value in texts.items():
        info.add_text(key, value)
    Image.new("RGB", (8, 8)).save(str(path), pnginfo=info)
    return str(path)


def text_chunks(path):
    with Image.open(path) as image:
        return dict(image.info)


def test_the_keys_carry_the_app_name():
    assert POSITIVE == "Teikr_OriginalPositiveTags"
    assert NEGATIVE == "Teikr_OriginalNegativeTags"
    assert (LEGACY_POSITIVE, LEGACY_NEGATIVE) == ("SDR_OriginalPositiveTags", "SDR_OriginalNegativeTags")


def test_written_tags_are_read_back(tmp_path):
    path = make_png(tmp_path / "new.png", prompt="{}")
    extractor = ImageDataExtractor()
    extractor.add_prompt_decomposition_to_exif(path, "a cat", "blurry")

    assert extractor.get_prompt_decomposition(path) == ("a cat", "blurry")
    chunks = text_chunks(path)
    assert chunks[POSITIVE] == "a cat"
    assert chunks["prompt"] == "{}"


def test_an_image_from_an_earlier_version_is_read(tmp_path):
    path = make_png(tmp_path / "old.png", **{LEGACY_POSITIVE: "a dog", LEGACY_NEGATIVE: "dark"})
    assert ImageDataExtractor().get_prompt_decomposition(path) == ("a dog", "dark")


def test_the_current_name_wins_over_a_legacy_one(tmp_path):
    path = make_png(tmp_path / "both.png", **{POSITIVE: "current", LEGACY_POSITIVE: "former"})
    assert ImageDataExtractor().get_prompt_decomposition(path) == ("current", None)


def test_writing_a_key_replaces_its_legacy_name(tmp_path):
    path = make_png(tmp_path / "old.png", **{LEGACY_POSITIVE: "a dog", LEGACY_NEGATIVE: "dark"})
    ImageDataExtractor().add_prompt_decomposition_to_exif(path, "a cat")

    chunks = text_chunks(path)
    assert chunks[POSITIVE] == "a cat"
    assert LEGACY_POSITIVE not in chunks
    # Not written in this call, so the legacy value stays and is still read.
    assert chunks[LEGACY_NEGATIVE] == "dark"
    assert ImageDataExtractor().get_prompt_decomposition(path) == ("a cat", "dark")
