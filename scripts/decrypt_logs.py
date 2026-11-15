"""Print or save the plaintext of the app's encrypted log files.

    python scripts/decrypt_logs.py                     most recent log, to stdout
    python scripts/decrypt_logs.py FILE [FILE ...]     the given files
    python scripts/decrypt_logs.py --all -o DIR        every log, as DIR/<name>.log

Must run as the user who wrote the logs: the key is derived from a passphrase in
that user's keyring (or MYPERSONALAPPLICATIONSSERVICELOGS_PASSPHRASE, if set).
Logs written under a former identity of the log key are read too: each record
is tried with the key of every identity that has a passphrase.
"""

import argparse
import os
import re
import struct
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cryptography.exceptions import InvalidTag  # noqa: E402

from lib.encryptor import StreamingLogCipher, get_log_cipher_key, peek_passphrase  # noqa: E402
from lib.logging_setup import (  # noqa: E402
    LEGACY_LOG_ENCRYPTION_APP_IDS,
    LOG_ENCRYPTION_SERVICE,
    LOG_FILE_SUFFIX,
    app_data_dir,
    log_key_identifier,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log_files() -> list[Path]:
    """Every encrypted log in the app's logs folder, oldest first."""
    return sorted(app_data_dir("logs", create=False).glob(f"*{LOG_FILE_SUFFIX}"))


def log_keys() -> list[bytes]:
    """The key of the current identity, then those of the former ones that have a
    passphrase; one and the same key once the passphrase has been copied."""
    keys = [get_log_cipher_key(LOG_ENCRYPTION_SERVICE, log_key_identifier())]
    for legacy in LEGACY_LOG_ENCRYPTION_APP_IDS:
        # Checked first, since deriving a key for an identity without a
        # passphrase would generate and store one.
        if peek_passphrase(LOG_ENCRYPTION_SERVICE, legacy):
            key = get_log_cipher_key(LOG_ENCRYPTION_SERVICE, legacy)
            if key not in keys:
                keys.append(key)
    return keys


def records(stream):
    """Yield the payload of each record; stop at a truncated trailing one."""
    while True:
        prefix = stream.read(4)
        if len(prefix) < 4:
            return
        length = struct.unpack(">I", prefix)[0]
        payload = stream.read(length)
        if len(payload) < length:
            return
        yield payload


def decrypt_record(keys: list[bytes], payload: bytes) -> bytes:
    for key in keys:
        try:
            return StreamingLogCipher.decrypt_record(key, payload)
        except InvalidTag:
            continue
    raise InvalidTag()


def decrypt(path: Path, strip_colors: bool, keys: list[bytes]):
    """Yield the lines of *path*; stop at the first record no key decrypts."""
    count = 0
    with open(path, "rb") as f:
        for payload in records(f):
            try:
                record = decrypt_record(keys, payload)
            except InvalidTag:
                print(f"{path.name}: record {count + 1} did not decrypt (wrong key, or the file "
                      "is corrupted); stopping here.", file=sys.stderr)
                return
            line = record.decode("utf-8", errors="replace")
            count += 1
            yield _ANSI.sub("", line) if strip_colors else line


def main() -> int:
    parser = argparse.ArgumentParser(description="Decrypt the app's encrypted log files.")
    parser.add_argument("files", nargs="*", type=Path, help="Log files to decrypt (default: the most recent).")
    parser.add_argument("--all", action="store_true", help="Decrypt every log in the logs folder.")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="Write each log to <dir>/<name>.log instead of printing it.")
    parser.add_argument("--keep-colors", action="store_true",
                        help="Keep the ANSI colour codes the log lines were written with.")
    args = parser.parse_args()

    paths = args.files or log_files()
    if not args.files and not args.all:
        paths = paths[-1:]
    if not paths:
        print(f"No {LOG_FILE_SUFFIX} files found in {app_data_dir('logs', create=False)}", file=sys.stderr)
        return 1

    keys = log_keys()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            out = args.output_dir / path.name[: -len(".enc")]
            with open(out, "w", encoding="utf-8") as f:
                for line in decrypt(path, not args.keep_colors, keys):
                    f.write(line + "\n")
            print(f"{path.name} -> {out}")
    else:
        strip = not (args.keep_colors or sys.stdout.isatty())
        for path in paths:
            if len(paths) > 1:
                print(f"===== {path.name} =====")
            for line in decrypt(path, strip, keys):
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
