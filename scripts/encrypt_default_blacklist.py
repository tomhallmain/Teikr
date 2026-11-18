"""Write the user's tag and model blacklists out as the encrypted default.

The lists written are the working lists in the app cache. They are only used
once the cache marks the blacklist as the user's own (set when it is stored
non-empty, imported or confirmed in the UI); until then the app loads the
shipped default, and re-encrypting that would change nothing, so the script
refuses.

Before writing, the existing file is copied to <file>.bak. After writing, the
file is decrypted again and compared entry by entry, field by field, with what
was written; on any difference or error the previous file is put back.

    python scripts/encrypt_default_blacklist.py --dry-run [--show-diff]
    python -m scripts.encrypt_default_blacklist

Exit codes: 0 written and verified, or a dry run that found nothing blocking;
1 refused (no working list, empty tag list); 2 an input could not be read;
3 verification or writing failed and the previous file was restored;
4 that restore failed too, and the target needs checking by hand.
"""

import argparse
import os
import shutil
import sys
from collections import Counter

# Before the project imports, so running the file directly works as well as -m.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from teikr.prompts import blacklist_state  # noqa: E402
from teikr.prompts.blacklist import Blacklist  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_UNREADABLE = 2
EXIT_RESTORED = 3
EXIT_RESTORE_FAILED = 4

#: Label -> getter for each list the default blacklist file carries.
LISTS = {
    "tag": Blacklist.get_items,
    "model": Blacklist.get_model_items,
}


def backup_path(target: str) -> str:
    return target + ".bak"


def _clear_lists() -> None:
    # A file written before model items were included leaves the model list
    # untouched on decryption, and that must not read as its contents.
    Blacklist.TAG_BLACKLIST = []
    Blacklist.MODEL_BLACKLIST = []


def _current_dicts() -> dict[str, list[dict]]:
    """Each list as written: every entry's to_dict(), in order, duplicates kept."""
    return {label: [item.to_dict() for item in getter()] for label, getter in LISTS.items()}


def read_default_file() -> dict[str, list[dict]]:
    """Decrypt the default blacklist file and return its lists."""
    _clear_lists()
    Blacklist.decrypt_blacklist()
    return _current_dicts()


def read_working_lists() -> dict[str, list[dict]]:
    """Load the working lists the same way the app does."""
    _clear_lists()
    blacklist_state.set_blacklist()
    return _current_dicts()


def _keyed(entries: list[dict]) -> dict[tuple[str, int], dict]:
    """Key each entry by its string and which occurrence of that string it is.

    The occurrence number keeps duplicate strings apart, so every entry is
    counted and the totals match what is written.
    """
    seen = Counter()
    keyed = {}
    for entry in entries:
        string = entry.get("string")
        keyed[(string, seen[string])] = entry
        seen[string] += 1
    return keyed


def compare(old: list[dict], new: list[dict]) -> dict[str, list]:
    before, after = _keyed(old), _keyed(new)
    return {
        "unchanged": [k for k in after if k in before and before[k] == after[k]],
        "changed": [k for k in after if k in before and before[k] != after[k]],
        "added": [k for k in after if k not in before],
        "removed": [k for k in before if k not in after],
        "_before": before,
        "_after": after,
    }


def duplicates(entries: list[dict]) -> dict[str, int]:
    """String -> number of entries carrying it, for strings on more than one."""
    counts = Counter(entry.get("string") for entry in entries)
    return {string: n for string, n in counts.items() if n > 1}


def _describe_key(key: tuple[str, int]) -> str:
    string, occurrence = key
    return f"{string!r}" + (f" (occurrence {occurrence + 1})" if occurrence else "")


def report(existing: dict[str, list[dict]], working: dict[str, list[dict]], show_diff: bool) -> None:
    for label in LISTS:
        old, new = existing[label], working[label]
        diff = compare(old, new)
        print(f"\n--- {label} blacklist: {len(new)} entries to write, "
              f"file holds {len(old)} ---")
        for heading in ("unchanged", "changed", "added", "removed"):
            print(f"  {heading:<10}: {len(diff[heading])}")

        dupes = duplicates(new)
        if dupes:
            extra = sum(n - 1 for n in dupes.values())
            print(f"  {len(dupes)} string(s) appear more than once "
                  f"({extra} extra entries, all of them written):")
            for string, n in sorted(dupes.items()):
                print(f"    {string!r} x{n}")

        if not show_diff:
            continue
        before, after = diff["_before"], diff["_after"]
        for heading, mark in (("changed", "~"), ("added", "+"), ("removed", "-")):
            keys = diff[heading]
            if not keys:
                continue
            print(f"\n  {label} entries {heading} ({len(keys)}):")
            for key in sorted(keys, key=lambda k: (str(k[0]), k[1])):
                print(f"    {mark} {_describe_key(key)}")
                if heading == "changed":
                    was, now = before[key], after[key]
                    for field in sorted(was.keys() | now.keys()):
                        if was.get(field) != now.get(field):
                            print(f"        {field}: {was.get(field)!r} -> {now.get(field)!r}")


def _restore(target: str, backup: str | None) -> bool:
    """Put the target back as it was before this run. True on success."""
    try:
        if backup is not None:
            shutil.copy2(backup, target)
            print(f"Restored the previous file from {backup}")
        elif os.path.exists(target):
            os.remove(target)
            print(f"Removed {target}: there was no file before this run")
        return True
    except Exception as e:
        print(f"RESTORE FAILED: {type(e).__name__}: {e}")
        print(f"Check {target} by hand" + (f"; the previous file is {backup}" if backup else ""))
        return False


def write_and_verify(target: str, working: dict[str, list[dict]]) -> int:
    """Back up, encrypt the lists currently loaded, read them back and compare."""
    backup = None
    if os.path.exists(target):
        backup = backup_path(target)
        try:
            shutil.copy2(target, backup)
        except Exception as e:
            print(f"Could not back up {target}, nothing written: {type(e).__name__}: {e}")
            return EXIT_UNREADABLE
        print(f"Backed up the previous file to {backup}")
    else:
        print("No previous file; no backup made")

    try:
        Blacklist.encrypt_blacklist()
        print(f"Encrypted the blacklist to {target}")
        read_back = read_default_file()
        mismatched = [label for label in LISTS if read_back[label] != working[label]]
    except Exception as e:
        print(f"FAILED writing or reading back: {type(e).__name__}: {e}")
        return EXIT_RESTORED if _restore(target, backup) else EXIT_RESTORE_FAILED

    if mismatched:
        for label in mismatched:
            diff = compare(working[label], read_back[label])
            print(f"FAILED verification of the {label} blacklist: wrote {len(working[label])} "
                  f"entries, read back {len(read_back[label])} ({len(diff['changed'])} changed, "
                  f"{len(diff['added'])} extra, {len(diff['removed'])} missing)")
        return EXIT_RESTORED if _restore(target, backup) else EXIT_RESTORE_FAILED

    counts = ", ".join(f"{len(working[label])} {label}" for label in LISTS)
    print(f"Verified by decrypting it back: {counts} entries, all fields equal")
    return EXIT_OK


def _tolerate_unencodable_output() -> None:
    # Blacklist strings can hold characters the console encoding (cp1252 on
    # Windows) cannot encode; escape them instead of raising mid-report.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass


def main(argv=None) -> int:
    _tolerate_unencodable_output()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Report what would change and write nothing, not even a backup.")
    parser.add_argument(
        "--show-diff", action="store_true",
        help="Also list every changed, added and removed entry. Can run to "
             "thousands of lines; redirect it to a file.")
    args = parser.parse_args(argv)

    target = Blacklist.DEFAULT_BLACKLIST_FILE_LOC

    if blacklist_state.is_in_default_state():
        print("Source: the shipped default. The app cache does not mark the blacklist "
              "as the user's own, so the app loads the existing default file and "
              "writing it again would change nothing. Nothing written.")
        print("To use your own list: edit, import or confirm it in the Blacklist "
              "window, which stores it as the working list, then run this again.")
        return EXIT_REFUSED

    if os.path.exists(target):
        try:
            existing = read_default_file()
        except Exception as e:
            print(f"Could not read the existing file {target}: {e}")
            print("Nothing written.")
            return EXIT_UNREADABLE
    else:
        print(f"No existing file at {target}")
        existing = {label: [] for label in LISTS}

    try:
        working = read_working_lists()
    except Exception as e:
        print(f"Could not load the working blacklist: {type(e).__name__}: {e}")
        return EXIT_UNREADABLE
    print("Source: the working blacklist in the app cache")

    report(existing, working, args.show_diff)
    print()

    if not working["tag"]:
        print("The tag blacklist is empty; refusing to write it. Nothing written.")
        return EXIT_REFUSED
    if not working["model"] and existing["model"]:
        print(f"Note: the model blacklist is empty; the {len(existing['model'])} model "
              f"entries in the existing file would be dropped.")

    if args.dry_run:
        print(f"Dry run: nothing written. {target} is unchanged.")
        return EXIT_OK

    return write_and_verify(target, working)


if __name__ == "__main__":
    sys.exit(main())
