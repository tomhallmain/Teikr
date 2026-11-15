"""The app identifier the encryption keys are filed under, after a rename.

Key material is filed per ``(Globals.SERVICE_NAME, app identifier)``: the key
store, which also holds the stored passwords, and the passphrase in the OS
keychain. After a rename an existing installation still has them under one of
``Globals.LEGACY_APP_IDENTIFIERS``. The identifier takes no part in the
cryptography, so the first lookup copies that material to
``Globals.APP_IDENTIFIER`` unchanged. The same keys then open every file
encrypted under the old name, backups included, without re-encrypting
anything, which also works for quantum keys in a process without OQS. The
stored password comes along inside the key store.

The legacy material is never deleted here. Until a copy has succeeded the
legacy identifier stays in use, so a failed copy only defers the migration.
"""

import os
import threading
from typing import Optional

import lib.encryptor as encryptor
from lib.logging_setup import get_logger
from lib.translations import _
from teikr.globals import Globals

logger = get_logger("persistence.key_identity")

_lock = threading.Lock()
_resolved: Optional[str] = None


def key_identifier() -> str:
    """The identifier to encrypt and decrypt with in this process.

    ``Globals.APP_IDENTIFIER`` when it has key material or nothing has any (a
    new installation). Otherwise the material of the first legacy identifier
    that has some is copied to it; if that fails, the legacy identifier.
    """
    global _resolved
    with _lock:
        if _resolved is None:
            _resolved = _resolve()
        return _resolved


def reset() -> None:
    """Forget the resolved identifier, so the next lookup resolves again."""
    global _resolved
    with _lock:
        _resolved = None


def _resolve() -> str:
    service, current = Globals.SERVICE_NAME, Globals.APP_IDENTIFIER
    if _has_key_material(service, current):
        return current
    for legacy in Globals.LEGACY_APP_IDENTIFIERS:
        if legacy == current or not _has_key_material(service, legacy):
            continue
        if _copy_key_material(service, legacy, current):
            logger.info(_(
                "Copied the encryption keys from '{0}' to '{1}'. The keys under "
                "'{0}' were left in place; remove them only once everything "
                "opens under '{1}'."
            ).format(legacy, current))
            return current
        return legacy
    return current


def _passphrase_from_environment(service: str) -> bool:
    # The variable answers for every identifier of the service, so there is no
    # stored passphrase to copy and none should be written.
    return f"{service.upper()}_PASSPHRASE" in os.environ


def _has_key_material(service: str, identifier: str) -> bool:
    """Whether *identifier* has keys of its own.

    ``has_prior_key_material`` counts a passphrase as evidence, and a
    passphrase from the environment is returned for every identifier, so with
    the variable set only a key store or the per-item keychain layout counts.
    """
    if not _passphrase_from_environment(service):
        return encryptor.has_prior_key_material(service, identifier)
    if encryptor.read_key_store(service, identifier) is not None:
        return True
    try:
        return bool(encryptor.keyring.get_password(service, encryptor.namespaced_key(identifier, "salt")))
    except Exception:
        return False


def _copy_key_material(service: str, legacy: str, current: str) -> bool:
    """Copy *legacy*'s key store and passphrase to *current*; True once read back.

    Only called while *current* has no key material, so undoing a partial copy
    removes nothing that existed before.
    """
    store = encryptor.read_key_store(service, legacy)
    if store is None:
        # Either still in the per-item keychain layout, which the encryptor
        # consolidates into a key store the next time it uses these keys, or
        # lost, which the encryptor reports rather than replacing the keys.
        logger.warning(_(
            "No key store was found for '{0}'; '{0}' stays in use, and its keys "
            "are copied to '{1}' at a start that finds one."
        ).format(legacy, current))
        return False
    from_environment = _passphrase_from_environment(service)
    passphrase = None if from_environment else encryptor.peek_passphrase(service, legacy)
    if not from_environment and not passphrase:
        logger.warning(_(
            "The passphrase for the encryption keys under '{0}' could not be read; "
            "'{0}' stays in use until a later start copies them to '{1}'."
        ).format(legacy, current))
        return False

    passphrase_key = encryptor.namespaced_key(current, "passphrase")
    try:
        # The passphrase first: a key store without it would make the
        # encryptor generate a new passphrase that cannot open the copy.
        if passphrase:
            encryptor.keyring.set_password(service, passphrase_key, passphrase)
        encryptor.write_key_store(service, current, store)
        encryptor.clear_key_store_cache(service, current)
        copied = encryptor.read_key_store(service, current)
        if copied != store or (passphrase and encryptor.peek_passphrase(service, current) != passphrase):
            raise RuntimeError("the copy did not read back as written")
    except Exception as e:
        logger.warning(_(
            "Could not copy the encryption keys from '{0}' to '{1}'; '{0}' stays "
            "in use until a later start succeeds: {2}"
        ).format(legacy, current, e))
        _undo_copy(service, current, passphrase_key, bool(passphrase))
        return False
    # Picked before the copy existed, from "no stored type"; read it again.
    encryptor.ENCRYPTOR_CLASSES.pop(encryptor._get_encryptor_key(service, current), None)
    return True


def _undo_copy(service: str, current: str, passphrase_key: str, wrote_passphrase: bool) -> None:
    encryptor.delete_key_store(service, current)
    if wrote_passphrase:
        try:
            encryptor.keyring.delete_password(service, passphrase_key)
        except Exception:
            pass
    encryptor.clear_key_store_cache(service, current)
