"""The keys of the encrypted files shipped in src/teikr/data.

Both files are encrypted with keys derived from ``Globals.SHIPPED_DATA_KEY_ID``,
which is kept apart from ``Globals.APP_IDENTIFIER``. These tests decrypt the
files as shipped, so that a change to either identifier cannot make them
unreadable unnoticed.
"""

import json

from lib.encryptor import symmetric_decrypt_data_from_file
from lib.utils import Utils
from teikr.globals import Globals
from teikr.prompts.blacklist import Blacklist
from teikr.prompts.concepts import Concepts


def test_the_key_seed_is_pinned():
    # The shipped files were encrypted under this value. Changing it needs both
    # files encrypted again, not just this constant.
    assert Globals.SHIPPED_DATA_KEY_ID == "sd_runner"


def test_the_shipped_default_blacklist_decrypts():
    encoded = symmetric_decrypt_data_from_file(
        Blacklist.DEFAULT_BLACKLIST_FILE_LOC,
        (Globals.SHIPPED_DATA_KEY_ID + "_blacklist").encode("utf-8"),
    )
    data = json.loads(Utils.postprocess_data_from_decryption(encoded))
    tags = data if isinstance(data, list) else data.get("tags", [])
    assert tags


def test_the_shipped_urban_dictionary_corpus_decrypts():
    assert Concepts.load_urban_dictionary_corpus(Concepts.URBAN_DICTIONARY_CORPUS_PATH)
