import logging
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from lib.custom_formatter import CustomFormatter
import lib.encryptor as encryptor
from lib.encryptor import encrypt_log_record, get_log_cipher_key, namespaced_key, peek_passphrase

#: The keyring identity the log key is derived under; a reader of the log
#: files must use the same two strings. The app's service name with a suffix:
#: PassphraseManager's env-var override is keyed on the service name alone, so
#: the test suite can pin this passphrase (MYPERSONALAPPLICATIONSSERVICELOGS_PASSPHRASE)
#: without also answering every passphrase lookup for the app's encrypted cache.
LOG_ENCRYPTION_SERVICE = "MyPersonalApplicationsServiceLogs"
LOG_ENCRYPTION_APP_ID = "teikr"
#: Identities the log key was filed under before, newest first. Never remove
#: an entry: the passphrase under one can be the only way to read the logs
#: written with it.
LEGACY_LOG_ENCRYPTION_APP_IDS = ("sd_runner",)
LOG_FILE_SUFFIX = ".log.enc"
#: The app data folder, under %APPDATA% on Windows and ~/.local/share elsewhere.
APP_DATA_FOLDER = "teikr"
#: Names the app data folder had before, newest first. The first time the
#: folder is resolved in a process, one found under such a name is moved to
#: APP_DATA_FOLDER.
LEGACY_APP_DATA_FOLDERS = ("sd_runner",)
#: Name of the lock that keeps a second instance from starting (a Windows
#: mutex ``<name>_SingleInstance``, else the lock file ``<name lower>.lock`` in
#: the temp folder). Fixed rather than derived from the display name, since
#: running versions must agree on it. Defined here because moving the app data
#: folder checks it, and that runs at import, before the entry point takes it.
SINGLE_INSTANCE_LOCK = "Teikr"
#: Lock names of earlier versions, held as well, so that an earlier version
#: and this one cannot run at the same time in either order. One can go once
#: no installation still starts a version that uses it.
LEGACY_SINGLE_INSTANCE_LOCKS = ("SDRunner",)
#: What a compiled build puts in the app data folder before any Python runs:
#: the onefile bootstrap unpacks there. A folder holding only this holds no
#: user data.
_BOOTSTRAP_ENTRIES = frozenset({"unpacked"})
#: The log file's suffix when the user has turned encryption off.
PLAINTEXT_LOG_FILE_SUFFIX = ".log"


def _lock_file(f) -> None:
    """Block until this process holds the OS lock on *f*'s first byte."""
    if sys.platform == 'win32':
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f) -> None:
    if sys.platform == 'win32':
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class EncryptedFileHandler(logging.Handler):
    """Appends each record to *filename* as one StreamingLogCipher record.

    The file is opened per record, so no handle is held between writes. Each
    append is made under an OS file lock as well as the handler's own: records
    are chained by their length prefixes, so a record torn by a concurrent
    writer would make every later one unreadable, and processes that log here
    without the single-instance lock (a second launch before it exits, the
    scripts) are separate writers.
    """

    def __init__(self, filename: Path, service_name: str, app_identifier: str):
        super().__init__()
        self.baseFilename = os.path.abspath(filename)
        self.key = get_log_cipher_key(service_name, app_identifier)
        self._write_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = encrypt_log_record(self.key, self.format(record).encode('utf-8'))
            with self._write_lock, open(self.baseFilename, 'ab') as f:
                _lock_file(f)
                try:
                    f.write(payload)
                    f.flush()
                finally:
                    _unlock_file(f)
        except Exception:
            self.handleError(record)


#: One handler for every logger: one key derivation per process, and one lock
#: in front of the file. False once creating it has failed, so each later
#: logger does not retry.
_shared_file_handler: logging.Handler | bool | None = None
#: Whether that handler writes the encrypted log. Loggers are created at import,
#: before config.json is read, so this starts on and the app turns it off
#: through set_log_file_encryption once the setting is known.
_encrypt_log_file: bool = True
#: The level set_logger_level last applied, given to loggers created after it:
#: modules imported lazily fetch their logger long after startup.
_log_level: int = logging.INFO
#: The app data folder this process uses, once resolved.
_app_data_root: Optional[Path] = None
_app_data_lock = threading.Lock()
#: (level, message) pairs from moving the app data folder and the log key.
#: Both run before any logger has a file, so they are logged by the first
#: logger that gets one.
_app_data_notices: list = []
#: The identity log_key_identifier resolved for this process.
_log_key_id: Optional[str] = None


def _log_file_path(encrypted: bool) -> Path:
    date_str: str = datetime.now().strftime("%Y-%m-%d")
    suffix: str = LOG_FILE_SUFFIX if encrypted else PLAINTEXT_LOG_FILE_SUFFIX
    return app_data_dir('logs') / f'sd_runner_{date_str}{suffix}'


def _get_file_handler(log_file: Path) -> Optional[logging.Handler]:
    """The shared file handler, or None if the log key is unavailable.

    Without a key the app logs to the console only. A plaintext file is not the
    fallback, since leaving one on disk is what encryption is there to prevent;
    plaintext is written only when the user has turned encryption off.
    """
    global _shared_file_handler
    if _shared_file_handler is None:
        if _encrypt_log_file:
            try:
                handler = EncryptedFileHandler(log_file, LOG_ENCRYPTION_SERVICE, log_key_identifier())
            except Exception as e:
                print(f"Log file disabled: could not derive the log encryption key: {e}", file=sys.stderr)
                _shared_file_handler = False
                return None
        else:
            handler = logging.FileHandler(log_file, encoding='utf-8', delay=True)
        handler.setLevel(logging.INFO)
        handler.setFormatter(CustomFormatter())
        _shared_file_handler = handler
    return _shared_file_handler or None


def log_key_identifier() -> str:
    """The identity the log key is derived under in this process.

    The key comes from the passphrase alone; the identity only names its
    keyring item. So when LOG_ENCRYPTION_APP_ID has no passphrase and a legacy
    identity has one, it is copied over, and the one key then reads the logs
    of either identity. The legacy item stays. If the copy fails, the legacy
    identity is used for this process.
    """
    global _log_key_id
    if _log_key_id is None:
        _log_key_id = _resolve_log_key_identifier()
    return _log_key_id


def _resolve_log_key_identifier() -> str:
    service, current = LOG_ENCRYPTION_SERVICE, LOG_ENCRYPTION_APP_ID
    # A passphrase from the environment answers for every identity.
    if f"{service.upper()}_PASSPHRASE" in os.environ or peek_passphrase(service, current):
        return current
    for legacy in LEGACY_LOG_ENCRYPTION_APP_IDS:
        passphrase = peek_passphrase(service, legacy)
        if not passphrase:
            continue
        item = namespaced_key(current, "passphrase")
        written = False
        try:
            encryptor.keyring.set_password(service, item, passphrase)
            written = True
            if encryptor.keyring.get_password(service, item) != passphrase:
                raise RuntimeError("the copy did not read back as written")
        except Exception as e:
            if written:
                try:
                    encryptor.keyring.delete_password(service, item)
                except Exception:
                    pass
            _app_data_notices.append((logging.WARNING,
                f"Could not copy the log key passphrase from '{legacy}' to '{current}'; "
                f"logs are written under '{legacy}' until a later start succeeds: {e}"))
            return legacy
        _app_data_notices.append((logging.INFO,
            f"Copied the log key passphrase from '{legacy}' to '{current}'. The one under "
            f"'{legacy}' was left in place; remove it only once every reader uses '{current}'."))
        return current
    return current


def set_log_file_encryption(enabled: bool) -> None:
    """Move every project logger to an encrypted or a plaintext log file.

    A no-op when the setting is unchanged. The handler it replaces is closed,
    and its level carries over to the new one.
    """
    global _shared_file_handler, _encrypt_log_file
    if enabled == _encrypt_log_file:
        return
    old = _shared_file_handler or None
    _encrypt_log_file = enabled
    _shared_file_handler = None
    new = _get_file_handler(_log_file_path(enabled))
    if new is not None and old is not None:
        new.setLevel(old.level)

    for name in list(logging.Logger.manager.loggerDict):
        if not name.startswith('sd_runner.'):
            continue
        logger = logging.getLogger(name)
        # Loggers get_logger never set up (placeholders, stray children) have
        # no handlers and are left alone.
        if not logger.handlers:
            continue
        if old is not None:
            logger.removeHandler(old)
        if new is not None:
            logger.addHandler(new)
    if old is not None:
        old.close()

    if not enabled and new is not None:
        get_logger("root").warning(f"Log encryption is off: logging to {new.baseFilename} in plaintext")


def _log_file_date(log_file: Path) -> datetime:
    """The date in ``sd_runner_YYYY-MM-DD.log[.enc]``, or the file's mtime."""
    try:
        return datetime.strptime(log_file.name.split('.')[0].split('_')[-1], '%Y-%m-%d')
    except (ValueError, IndexError):
        return datetime.fromtimestamp(log_file.stat().st_mtime)


def _cleanup_old_logs(log_dir: Path, logger: logging.Logger) -> None:
    """
    Clean up log files that are older than 30 days if there are more than 10 log files.

    Plaintext ``.log`` files written before log encryption are cleaned up by
    the same rule as the encrypted ones.

    Args:
        log_dir: Path object pointing to the directory containing log files
        logger: Logger instance to use for logging cleanup operations
    """
    try:
        log_files: List[Path] = (
            list(log_dir.glob('sd_runner_*.log')) + list(log_dir.glob(f'sd_runner_*{LOG_FILE_SUFFIX}'))
        )
        if len(log_files) <= 10:
            return

        current_time: datetime = datetime.now()
        cutoff_date: datetime = current_time - timedelta(days=30)

        for log_file in log_files:
            if _log_file_date(log_file) < cutoff_date:
                log_file.unlink()
                logger.debug(f"Deleted old log file: {log_file}")
    except Exception as e:
        logger.error(f"Error cleaning up old log files: {e}")


def app_data_dir(*parts: str, create: bool = True) -> Path:
    """Return ``<app data>/<APP_DATA_FOLDER>/<parts...>``, creating it unless
    ``create`` is False.

    The app data base is %APPDATA% on Windows and ~/.local/share elsewhere.
    Logs, the user config and the blacklist filter cache all live under it.
    TEIKR_APP_DATA_DIR, if set, replaces that folder as a whole and
    suppresses the move from a legacy folder; the test suite sets it before
    this module is first imported.
    """
    override: str | None = os.getenv('TEIKR_APP_DATA_DIR')
    if override:
        root: Path = Path(override)
    else:
        root = _resolved_app_data_root()
    path: Path = root.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _app_data_base() -> Path:
    return Path(os.getenv('APPDATA') if sys.platform == 'win32' else os.path.expanduser('~/.local/share'))


def _resolved_app_data_root() -> Path:
    global _app_data_root
    with _app_data_lock:
        if _app_data_root is None:
            _app_data_root = migrate_app_data_folder(_app_data_base())
        return _app_data_root


def _entries(folder: Path) -> set:
    try:
        return set(os.listdir(folder))
    except OSError:
        return set()


def _running_instance() -> Optional[str]:
    """The name of a single-instance lock some process holds, else None.

    Only looks: nothing is taken. A process starting between this check and
    the move is not seen.
    """
    for name in (SINGLE_INSTANCE_LOCK,) + LEGACY_SINGLE_INSTANCE_LOCKS:
        if _single_instance_lock_held(name):
            return name
    return None


def _single_instance_lock_held(name: str) -> bool:
    """Whether the lock *name*, as Utils.check_single_instance takes it, is held."""
    if sys.platform == 'win32' and _mutex_exists(f"{name}_SingleInstance"):
        return True
    return _lock_file_held(os.path.join(tempfile.gettempdir(), f"{name.lower()}.lock"))


def _mutex_exists(mutex_name: str) -> bool:
    try:
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenMutexW.restype = ctypes.c_void_p
        synchronize = 0x00100000
        handle = kernel32.OpenMutexW(synchronize, False, mutex_name)
    except Exception:
        return False
    if not handle:
        return False
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return True


def _lock_file_held(path: str) -> bool:
    """Whether another process holds the lock on *path*'s first byte."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        # No lock file: nobody holds it.
        return False
    try:
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except Exception:
        return False
    finally:
        os.close(fd)


def migrate_app_data_folder(base: Path) -> Path:
    """Move a legacy app data folder under *base* to APP_DATA_FOLDER; return
    the folder to use.

    The whole folder is renamed when the new one does not exist. A build has
    already unpacked into the new one, so when it holds nothing else, the
    legacy folder's entries are moved into it one by one, all or none. When
    both hold data, the new one is used and nothing is merged. Whenever a move
    fails, the legacy folder stays in use for this process and the next start
    tries again.
    """
    new = base / APP_DATA_FOLDER
    for name in LEGACY_APP_DATA_FOLDERS:
        old = base / name
        if not old.is_dir():
            continue
        old_entries = _entries(old) - _BOOTSTRAP_ENTRIES
        if not old_entries:
            # Only an earlier build's unpack folders: the build's cleanup
            # removes them, and the folder with them once it is empty.
            continue
        # The move runs at import, before this process checks the
        # single-instance lock, so it must not pull the folder away from an
        # instance that is running from it.
        running = _running_instance()
        if running:
            _app_data_notices.append((logging.INFO,
                f"An instance ({running}) is running; {old} stays in use, and is moved to {new} "
                f"at a start while none is running"))
            return old
        if not new.exists():
            try:
                os.rename(old, new)
            except OSError as e:
                _app_data_notices.append((logging.WARNING,
                    f"Could not move the app data folder {old} to {new}; using {old} until a later start: {e}"))
                return old
            _app_data_notices.append((logging.INFO, f"Moved the app data folder {old} to {new}"))
            return new
        if _entries(new) - _BOOTSTRAP_ENTRIES:
            _app_data_notices.append((logging.WARNING,
                f"Both {old} and {new} hold app data; using {new}. {old} was left as it is."))
            return new
        moved: list = []
        try:
            for entry in sorted(old_entries):
                os.rename(old / entry, new / entry)
                moved.append(entry)
        except OSError as e:
            stuck = []
            for entry in reversed(moved):
                try:
                    os.rename(new / entry, old / entry)
                except OSError:
                    stuck.append(entry)
            if stuck:
                _app_data_notices.append((logging.ERROR,
                    f"Moving the app data from {old} to {new} failed ({e}), and these could not be "
                    f"moved back: {', '.join(sorted(stuck))}. They are in {new}; move them to {old} by hand."))
            else:
                _app_data_notices.append((logging.WARNING,
                    f"Could not move the app data from {old} to {new}; using {old} until a later start: {e}"))
            return old
        _app_data_notices.append((logging.INFO, f"Moved the app data from {old} to {new}"))
        return new
    return new


def relocate_legacy_app_data_path(path: str, root: Optional[Path] = None) -> str:
    """*path* moved into the app data folder in use, if it points into a legacy
    app data folder beside it and the same place exists in the new one;
    otherwise *path* unchanged.

    *root* is the folder in use, ``app_data_dir()`` by default. When that is a
    legacy folder (a move failed) or TEIKR_APP_DATA_DIR replaces it, nothing is
    relocated. A place that exists only in the legacy folder (both held data,
    so nothing was moved) keeps its path.
    """
    if root is None:
        if os.getenv('TEIKR_APP_DATA_DIR'):
            return path
        root = _resolved_app_data_root()
    if root.name != APP_DATA_FOLDER:
        return path
    original = os.path.normpath(path)
    # Compared case-folded on Windows; the rest keeps the path's own case.
    target = os.path.normcase(original)
    for name in LEGACY_APP_DATA_FOLDERS:
        legacy = os.path.normcase(os.path.normpath(root.parent / name))
        if target == legacy:
            relocated = root
        elif target.startswith(legacy + os.sep):
            relocated = root / original[len(legacy) + 1:]
        else:
            continue
        if relocated.exists():
            return os.path.normpath(str(relocated))
    return path


def _log_app_data_notices() -> None:
    if not _app_data_notices:
        return
    notices = list(_app_data_notices)
    _app_data_notices.clear()
    notice_logger = get_logger("app_data")
    for level, message in notices:
        notice_logger.log(level, message)


def adopt_legacy_file(legacy_path: str, target_path: str) -> None:
    """Move ``legacy_path`` to ``target_path``, carrying a file over from a
    previous storage location.

    An existing target is replaced: a compiled build started before the
    checkout writes a default config and an empty cache there, which must not
    keep the user's data from moving in. The exception is a legacy file newer
    than the target, which means something still writes the old location; both
    are then left in place with a warning. On failure the legacy file stays.
    """
    if not os.path.isfile(legacy_path):
        return
    logger = get_logger("root")
    if os.path.isdir(target_path):
        logger.warning(f"Not moving {legacy_path}: {target_path} is a folder")
        return
    if os.path.exists(target_path) and os.path.getmtime(legacy_path) > os.path.getmtime(target_path):
        logger.warning(f"Not moving {legacy_path}: it is newer than {target_path}; keeping both")
        return
    try:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        try:
            os.replace(legacy_path, target_path)
        except OSError:
            # os.replace cannot cross drives (a repo on D:, %APPDATA% on C:).
            shutil.copy2(legacy_path, target_path)
            os.remove(legacy_path)
        logger.info(f"Moved {legacy_path} to {target_path}")
    except OSError as e:
        logger.error(f"Failed to move {legacy_path} to {target_path}: {e}")


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        module_name: The name of the module requesting the logger
        
    Returns:
        A configured logger instance for the module
    """
    # Create logger with module name
    logger: logging.Logger = logging.getLogger(f"sd_runner.{module_name}")
    logger.setLevel(_log_level)
    logger.propagate = False

    # If handlers are already set up, return the logger
    if logger.handlers:
        return logger

    # create console handler with a higher log level
    ch: logging.StreamHandler = logging.StreamHandler()
    ch.setLevel(_log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    log_dir: Path = app_data_dir('logs')

    # Clean up old logs before creating new one
    _cleanup_old_logs(log_dir, logger)

    fh = _get_file_handler(_log_file_path(_encrypt_log_file))
    if fh is not None:
        logger.addHandler(fh)

    _log_app_data_notices()
    return logger

def set_logger_level(debug: bool) -> None:
    """
    Set the logger level to DEBUG if debug is True, otherwise set it to INFO.
    This updates all existing loggers in the sd_runner hierarchy and their handlers.
    """
    global _log_level
    level = logging.DEBUG if debug else logging.INFO
    _log_level = level

    # Update all existing loggers in the sd_runner hierarchy
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('sd_runner'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            # Also update all handlers for this logger
            for handler in logger.handlers:
                handler.setLevel(level)
    
    # Also update the root logger for backward compatibility
    root_logger = get_logger("root")
    root_logger.setLevel(level)
    # Update all handlers for root logger
    for handler in root_logger.handlers:
        handler.setLevel(level)

# Initialize root logger for backward compatibility
root_logger: logging.Logger = get_logger("root") 
