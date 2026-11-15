#!/usr/bin/env python3
"""
Extract candidate concepts from arbitrary documents and report which ones the
repository's concept files do not have yet.

Candidates are 1..N-word phrases that neither start nor end on a stopword. Each
one is matched against the known concepts in tiers, cheapest first; the first
tier that claims it decides:

  exact     case-insensitive equality, whitespace collapsed
  variant   equality after folding ``_``/``-`` to spaces, dropping punctuation
            and diacritics, and for English reducing simple plurals
  fuzzy     spelling variants: character-trigram candidates from an inverted
            index (prefix-filtered, so only rare trigrams are probed), verified
            with difflib's ratio
  semantic  optional: cosine similarity of word vectors (GloVe, word2vec or
            fastText text format) or of a CLIP/SigLIP text encoder; claims a
            candidate only when --semantic-threshold is given

What no tier claims is reported as new, with a suggested target file: voted by
the nearest semantic neighbours when vectors are loaded, else the file that
holds the phrase's last word, which is the head noun of an English noun phrase.
Blacklisted candidates are withheld.

The concept set is picked by the documents' language: every folder in the repo
root with a ``concept_set.json`` (``concepts``, ``Konzepte``) is a choice, its
manifest names the language, and the documents are detected among those
languages with Lingua, or by their share of stopwords where Lingua is missing.
--language overrides the detection, --concepts-dir the choice of set. The
stopwords and plural rules follow the language chosen.

Exact, variant and fuzzy matching use only the standard library. The semantic
tier needs numpy, and --clip also needs torch with either the ``clip`` package
or ``transformers`` (model in the local Hugging Face cache).

Readable inputs: plain text of any extension, HTML/XHTML, EPUB, and PDF when
``pdftotext`` is on PATH. Directories are searched recursively.

    python scripts/extract_new_concepts.py notes.txt book.epub
    python scripts/extract_new_concepts.py some_dir/ --max-ngram 2 --min-count 3
    python scripts/extract_new_concepts.py page.html --vectors glove.6B.300d.txt
    python scripts/extract_new_concepts.py doc.txt --clip ViT-B/32 --export-dir out/
    python scripts/extract_new_concepts.py artikel.txt --language de

Files written by --export-dir hold one concept per line, sorted, ready for the
Import button of the concept editor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teikr.prompts.language_codes import normalize_language_code  # noqa: E402


# Files that hold concepts but are not categories a new concept should be filed
# under. They still count as known.
NON_TARGET_FILES = frozenset({
    "dictionary.txt",
    "negatives.txt",
    "exclusionary_concepts.txt",
    "hard_concepts.txt",
    "boring_concepts.txt",
})
NON_TARGET_PREFIXES = ("nsfw_", "nsfl_")
DICTIONARY_FILE = "dictionary.txt"

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each either else ever every few for from further had has have having he
her here hers herself him himself his how however i if in into is it its itself
just let like may me might more most much must my myself neither no nor not now
of off often on once only or other others our ours ourselves out over own per
quite rather really said same say says shall she should since so some still
such than that the their theirs them themselves then there these they this
those though through thus to too under until up upon us very was we were what
whatever when where whether which while who whom whose why will with within
without would yet you your yours yourself yourselves one ones two also
""".split())

GERMAN_STOPWORDS = frozenset("""
aber alle allem allen aller alles als also am an ander andere anderem anderen
anderer anderes anders auch auf aus bei beim bin bis bist da damit dann das
dass dein deine deinem deinen deiner dem den denn der des dessen dich die dies
diese diesem diesen dieser dieses dir doch dort du durch ein eine einem einen
einer eines einige einigen einiger er es etwas euch euer eure für gegen gewesen
hab habe haben hat hatte hatten hier hin hinter ich ihm ihn ihnen ihr ihre ihrem
ihren ihrer im in indem ins ist ja jede jedem jeden jeder jedes jene jenem jenen
jener jenes jetzt kann kein keine keinem keinen keiner können konnte man manche
mein meine meinem meinen meiner mich mir mit muss musste nach nicht nichts noch
nun nur ob oder ohne sehr sein seine seinem seinen seiner seit sich sie sind so
solche soll sollte sondern sonst über um und uns unser unsere unter viel vom von
vor war waren warum was weil weiter welche welchem welchen welcher welches wenn
wer werde werden wie wieder will wir wird wo wollen worden wurde wurden zu zum
zur zwar zwei zwischen
""".split())

#: Built-in stopwords per primary language subtag. Another language needs
#: --stopwords to give phrases sensible boundaries.
STOPWORDS_BY_LANGUAGE = {"en": STOPWORDS, "de": GERMAN_STOPWORDS}


# ---------------------------------------------------------------------------
# Reading documents
# ---------------------------------------------------------------------------

class _HTMLText(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript"})
    _BLOCK = frozenset({
        "p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "aside",
        "blockquote", "dd", "dt", "pre", "header", "footer", "figcaption",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _HTMLText()
    parser.feed(markup)
    parser.close()
    return "".join(parser.parts)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _epub_to_text(path: Path) -> str:
    parts = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith((".xhtml", ".html", ".htm")):
                parts.append(html_to_text(_decode(zf.read(name))))
    return "\n\n".join(parts)


def _pdf_to_text(path: Path) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("reading PDF needs pdftotext (poppler-utils) on PATH")
    result = subprocess.run(
        [exe, "-enc", "UTF-8", str(path), "-"],
        capture_output=True, stdin=subprocess.DEVNULL, check=True,
    )
    return _decode(result.stdout)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _epub_to_text(path)
    if suffix == ".pdf":
        return _pdf_to_text(path)
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ValueError("binary file")
    text = _decode(raw)
    if suffix in (".html", ".htm", ".xhtml", ".xml"):
        return html_to_text(text)
    return text


def iter_document_paths(paths: list[Path]):
    for p in paths:
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and not any(part.startswith(".") for part in sub.relative_to(p).parts):
                    yield sub
        else:
            yield p


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s']+")
_WORD = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")


def exact_key(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _singular(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def variant_key(text: str, language: str = "en") -> str:
    """The folded form of *text*; plurals are reduced only for English."""
    t = _strip_diacritics(unicodedata.normalize("NFKC", text)).casefold()
    t = t.replace("’", "'").replace("_", " ").replace("-", " ")
    t = _NON_WORD.sub(" ", t)
    english = language == "en"
    words = []
    for w in t.split():
        if english:
            w = w.removesuffix("'s")
        w = w.strip("'")
        if w:
            words.append(_singular(w) if english else w)
    return " ".join(words)


def words_of(text: str) -> list[str]:
    return [w.casefold() for w in _WORD.findall(text.replace("’", "'"))]


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

# Phrases never span these: blank lines, sentence and clause punctuation,
# brackets, quotes, slashes, dashes, "--" and a hyphen with space around it.
_SEGMENT_BREAK = re.compile(
    r"\n[ \t]*\n|[.!?;:,()\[\]{}<>\"“”«»„…|/\\•·–—]+|-{2,}|\s-+\s"
)
# A token is a run of word characters with inner apostrophes or hyphens. One
# holding a digit or underscore is not a word but still breaks adjacency, so
# "red 3 apples" does not yield "red apples".
_TOKEN = re.compile(r"[\w'’-]+")


@dataclass
class Candidate:
    key: str
    n_words: int
    count: int = 0
    surfaces: Counter = field(default_factory=Counter)
    documents: set = field(default_factory=set)
    tier: str = "new"
    match: str | None = None
    match_file: str | None = None
    score: float | None = None
    target: str | None = None
    target_via: str | None = None
    neighbours: list = field(default_factory=list)

    @property
    def surface(self) -> str:
        # Concept files are mostly lower case: prefer that form when the
        # document used it, else the most frequent spelling.
        for form, _count in self.surfaces.most_common():
            if form == form.lower():
                return form
        return self.surfaces.most_common(1)[0][0]

    def to_dict(self) -> dict:
        return {
            "concept": self.surface,
            "count": self.count,
            "words": self.n_words,
            "tier": self.tier,
            "match": self.match,
            "match_file": self.match_file,
            "score": None if self.score is None else round(self.score, 4),
            "target": self.target,
            "target_via": self.target_via,
            "neighbours": self.neighbours,
            "documents": sorted(self.documents),
        }


def _word_runs(text: str):
    text = unicodedata.normalize("NFKC", text).replace("’", "'")
    for segment in _SEGMENT_BREAK.split(text):
        run: list[str] = []
        for m in _TOKEN.finditer(segment):
            tok = m.group().strip("'-")
            if not tok or any(c.isdigit() or c == "_" for c in tok):
                if run:
                    yield run
                run = []
            else:
                run.append(tok)
        if run:
            yield run


def extract_candidates(
    documents: list[tuple[str, str]],
    *,
    max_ngram: int,
    min_chars: int,
    stopwords: frozenset[str],
) -> tuple[dict[str, Candidate], int]:
    """Count every admissible n-gram across *documents* ``(label, text)``."""
    cands: dict[str, Candidate] = {}
    total_words = 0
    for label, text in documents:
        for run in _word_runs(text):
            total_words += len(run)
            folded = [w.casefold() for w in run]
            for i in range(len(run)):
                if folded[i] in stopwords:
                    continue
                for n in range(1, max_ngram + 1):
                    j = i + n
                    if j > len(run):
                        break
                    if folded[j - 1] in stopwords:
                        continue
                    surface = " ".join(run[i:j])
                    if len(surface) < min_chars:
                        continue
                    key = " ".join(folded[i:j])
                    c = cands.get(key)
                    if c is None:
                        c = cands[key] = Candidate(key=key, n_words=n)
                    c.count += 1
                    c.surfaces[surface] += 1
                    c.documents.add(label)
    return cands, total_words


def drop_subsumed(cands: dict[str, Candidate], min_count: int) -> int:
    """Drop phrases that only ever occur inside one longer phrase.

    When "golden retriever" occurs exactly as often as "golden retriever
    puppy", every occurrence is part of the longer phrase, so only that one is
    kept.
    """
    subsumed = set()
    for c in cands.values():
        if c.n_words < 2 or c.count < min_count:
            continue
        words = c.key.split(" ")
        for sub in (" ".join(words[:-1]), " ".join(words[1:])):
            other = cands.get(sub)
            if other is not None and other.count == c.count:
                subsumed.add(sub)
    for key in subsumed:
        del cands[key]
    return len(subsumed)


# ---------------------------------------------------------------------------
# Known concepts
# ---------------------------------------------------------------------------

def _parse_concept_line(line: str) -> str:
    """The concept on *line*, parsed the way the concepts loader does."""
    return line.split("#", 1)[0].strip()


def is_target_file(label: str) -> bool:
    name = label.rsplit("/", 1)[-1]
    return name not in NON_TARGET_FILES and not name.startswith(NON_TARGET_PREFIXES)


def discover_concept_files(concepts_dir: Path, include_dictionary: bool) -> list[tuple[str, Path]]:
    files = []
    for directory in (concepts_dir, concepts_dir / "_"):
        if not directory.is_dir():
            continue
        for p in sorted(directory.glob("*.txt")):
            label = p.relative_to(concepts_dir).as_posix()
            if label == DICTIONARY_FILE and not include_dictionary:
                continue
            files.append((label, p))
    return files


class FuzzyIndex:
    """Nearest key by difflib ratio, over an inverted index of padded trigrams.

    A key whose trigram Jaccard similarity to the query is at least ``floor``
    shares at least ``floor * |q|`` trigrams with it, so it must contain one of
    the query's ``|q| - ceil(floor * |q|) + 1`` rarest trigrams. Only those
    postings are read, and the Jaccard bound is checked before the ratio.
    """

    def __init__(self, keys: list[str], jaccard_floor: float = 0.5) -> None:
        self.keys = keys
        self.floor = jaccard_floor
        self.grams = [self._trigrams(k) for k in keys]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, grams in enumerate(self.grams):
            for g in grams:
                self.postings[g].append(i)

    @staticmethod
    def _trigrams(key: str) -> frozenset[str]:
        s = f" {key} "
        return frozenset(s[i:i + 3] for i in range(len(s) - 2))

    def best(self, key: str, threshold: float) -> tuple[float, int | None]:
        q = self._trigrams(key)
        nq = len(q)
        if not nq:
            return 0.0, None
        ordered = sorted(q, key=lambda g: len(self.postings.get(g, ())))
        probe = ordered[: nq - math.ceil(self.floor * nq) + 1]
        lo, hi = self.floor * nq, nq / self.floor
        seen: set[int] = set()
        best_ratio, best_i = 0.0, None
        for g in probe:
            for i in self.postings.get(g, ()):
                if i in seen:
                    continue
                seen.add(i)
                r = self.grams[i]
                nr = len(r)
                if nr < lo or nr > hi:
                    continue
                inter = len(q & r)
                if inter / (nq + nr - inter) < self.floor:
                    continue
                other = self.keys[i]
                sm = SequenceMatcher(None, key, other, autojunk=False)
                if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                    continue
                ratio = sm.ratio()
                if ratio >= threshold and ratio > best_ratio:
                    best_ratio, best_i = ratio, i
        return best_ratio, best_i


class KnownConcepts:
    def __init__(
        self, files: list[tuple[str, Path]], extra_known: list[Path], max_words: int, language: str = "en",
    ) -> None:
        self.language = language
        self.entries: list[tuple[str, str]] = []          # (concept, file label)
        self.exact: dict[str, int] = {}
        self.variant: dict[str, int] = {}
        self.head_files: dict[str, Counter] = defaultdict(Counter)
        self.file_count = 0
        sources = [(label, p) for label, p in files] + [(f"[known] {p.name}", p) for p in extra_known]
        for label, path in sources:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as e:
                print(f"Warning: could not read {path}: {e}", file=sys.stderr)
                continue
            self.file_count += 1
            target = is_target_file(label) and not label.startswith("[known]")
            for line in lines:
                concept = _parse_concept_line(line)
                if not concept:
                    continue
                idx = len(self.entries)
                self.entries.append((concept, label))
                self.exact.setdefault(exact_key(concept), idx)
                if "[[" in concept or "$$" in concept:
                    continue  # template line; its raw text is all that is matched
                vkey = variant_key(concept, language)
                if vkey:
                    self.variant.setdefault(vkey, idx)
                    words = vkey.split(" ")
                    if target and len(words) <= 3:
                        self.head_files[words[-1]][label] += 1
        # Entries far longer than any candidate cannot pass the fuzzy length
        # bound, so they are left out of the index.
        self._fuzzy_keys = [k for k in self.variant if k.count(" ") < 2 * max_words]
        self._fuzzy: FuzzyIndex | None = None

    def fuzzy(self) -> FuzzyIndex:
        if self._fuzzy is None:
            self._fuzzy = FuzzyIndex(self._fuzzy_keys)
        return self._fuzzy

    def suggest_by_head(self, vkey: str, allowed: set[str]) -> str | None:
        files = self.head_files.get(vkey.split(" ")[-1])
        for label, _count in (files.most_common() if files else ()):
            if not allowed or label in allowed:
                return label
        return None

    def neighbour_pool(self, max_words: int) -> list[tuple[str, str]]:
        """Distinct short concepts from target files, for semantic neighbours."""
        seen = set()
        pool = []
        for concept, label in self.entries:
            if not is_target_file(label) or label.startswith("[known]"):
                continue
            if "[[" in concept or "$$" in concept or concept.count(" ") >= 2 * max_words:
                continue
            k = exact_key(concept)
            if k in seen:
                continue
            seen.add(k)
            pool.append((concept, label))
        return pool


# ---------------------------------------------------------------------------
# Concept sets and language
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "concept_set.json"
#: Text read per document for detection; more adds time, not accuracy.
DETECTION_SAMPLE_CHARS = 20_000
#: Below this share of stopwords no language is assumed by the fallback.
MIN_STOPWORD_SHARE = 0.05


@dataclass(frozen=True)
class ConceptSet:
    path: Path
    language: str  # primary subtag; "en" when the manifest names none


def read_concept_set(path: Path) -> ConceptSet:
    """The set at *path*. A missing or unreadable manifest means English."""
    try:
        manifest = json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    code = manifest.get("language") if isinstance(manifest, dict) else None
    return ConceptSet(path, normalize_language_code(code) or "en")


def discover_concept_sets(root: Path, extra: list[Path]) -> list[ConceptSet]:
    """Every folder under *root* with a manifest, then the *extra* folders."""
    found = [read_concept_set(d) for d in sorted(root.iterdir())
             if d.is_dir() and (d / MANIFEST_FILENAME).is_file()]
    seen = {s.path for s in found}
    for d in extra:
        if d not in seen:
            seen.add(d)
            found.append(read_concept_set(d))
    return found


@dataclass
class LanguageGuess:
    language: str | None
    method: str
    scores: dict[str, float] = field(default_factory=dict)
    per_document: dict[str, str | None] = field(default_factory=dict)


def _lingua_scorer(languages: frozenset[str]):
    """A ``text -> {code: confidence}`` function over *languages*, or None without Lingua."""
    try:
        from lingua import IsoCode639_1, LanguageDetectorBuilder  # noqa: PLC0415
    except ImportError:
        return None
    codes = [getattr(IsoCode639_1, c.upper()) for c in sorted(languages) if hasattr(IsoCode639_1, c.upper())]
    if len(codes) < len(languages):
        return None  # a language Lingua cannot detect would never be chosen
    detector = LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()

    def score(text: str) -> dict[str, float]:
        return {v.language.iso_code_639_1.name.lower(): v.value
                for v in detector.compute_language_confidence_values(text)}

    return score


def _stopword_scorer(languages: frozenset[str]):
    """Share of each language's built-in stopwords among the words of a text."""
    lists = {code: STOPWORDS_BY_LANGUAGE[code] for code in languages if code in STOPWORDS_BY_LANGUAGE}
    if len(lists) < len(languages):
        return None

    def score(text: str) -> dict[str, float]:
        words = words_of(text)
        if not words:
            return {}
        return {code: sum(w in stop for w in words) / len(words) for code, stop in lists.items()}

    return score


def detect_language(documents: list[tuple[str, str]], languages: frozenset[str]) -> LanguageGuess:
    """The language of *documents* among *languages*, weighting each by its sample length."""
    if len(languages) == 1:
        return LanguageGuess(next(iter(languages)), "only set")
    method, scorer = "lingua", _lingua_scorer(languages)
    if scorer is None:
        method, scorer = "stopwords", _stopword_scorer(languages)
    if scorer is None:
        return LanguageGuess(None, "undetectable")
    totals: Counter = Counter()
    weight = 0
    per_document: dict[str, str | None] = {}
    for label, text in documents:
        sample = text[:DETECTION_SAMPLE_CHARS]
        scores = scorer(sample)
        best = max(scores.items(), key=lambda kv: kv[1], default=(None, 0.0))
        if method == "stopwords" and best[1] < MIN_STOPWORD_SHARE:
            best = (None, 0.0)
        per_document[label] = best[0]
        for code, value in scores.items():
            totals[code] += value * len(sample)
        weight += len(sample)
    scores = {code: value / weight for code, value in totals.items()} if weight else {}
    language = max(scores, key=scores.get) if scores else None
    if method == "stopwords" and language is not None and scores[language] < MIN_STOPWORD_SHARE:
        language = None
    return LanguageGuess(language, method, scores, per_document)


# ---------------------------------------------------------------------------
# Semantic encoders (optional; numpy imported only here)
# ---------------------------------------------------------------------------

def _normalise_rows(np, mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class WordVectorEncoder:
    """Phrase vector = mean of its content words' vectors.

    Reads a GloVe, word2vec or fastText text file (optionally ``.gz``) in one
    pass, keeping only the words in *vocab*. A phrase stored as one token with
    underscores (``new_york``, as in word2vec releases) is used when present.
    """

    cacheable = False

    def __init__(self, np, path: Path, vocab: set[str], stopwords: frozenset[str]) -> None:
        self.np = np
        self.stopwords = stopwords
        self.backend_id = f"vectors:{path.name}"
        self.vectors: dict[str, object] = {}
        self.dim = 0
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f):
                word, _sep, rest = line.rstrip().partition(" ")
                if lineno == 0 and word.isdigit() and rest.strip().isdigit():
                    continue  # word2vec header: "<count> <dim>"
                w = word.casefold()
                if w not in vocab or w in self.vectors:
                    continue
                try:
                    vec = np.array(rest.split(), dtype=np.float32)
                except ValueError:
                    continue
                if not self.dim:
                    self.dim = vec.size
                if vec.size == self.dim:
                    self.vectors[w] = vec
        if not self.dim:
            raise RuntimeError(f"no usable vectors for this vocabulary in {path}")

    @staticmethod
    def vocabulary(phrases) -> set[str]:
        vocab = set()
        for p in phrases:
            words = words_of(p)
            vocab.update(words)
            if len(words) > 1:
                vocab.add("_".join(words))
        return vocab

    def encode(self, phrases: list[str]):
        np = self.np
        out = np.zeros((len(phrases), self.dim), dtype=np.float32)
        for i, phrase in enumerate(phrases):
            words = words_of(phrase)
            joined = self.vectors.get("_".join(words)) if len(words) > 1 else None
            if joined is not None:
                out[i] = joined
                continue
            content = [w for w in words if w not in self.stopwords] or words
            vecs = [self.vectors[w] for w in content if w in self.vectors]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return _normalise_rows(np, out)


class ClipTextEncoder:
    """CLIP or SigLIP text embeddings.

    A ``clip`` package model name (``ViT-B/32``) or ``.pt`` file goes through
    that package; anything else is a Hugging Face id or folder, loaded from the
    local cache only so that nothing is downloaded silently.
    """

    cacheable = True

    def __init__(self, np, name: str, device: str | None, batch_size: int) -> None:
        import torch  # noqa: PLC0415

        self.np = np
        self.torch = torch
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend_id = f"clip:{name}"
        self._clip = None
        try:
            import clip  # noqa: PLC0415
            if os.path.isfile(name) or name in clip.available_models():
                from lib.clip_loader import load_clip  # noqa: PLC0415
                self._clip = clip
                self._model, _preprocess = load_clip(clip, torch, name, self.device)
        except ImportError:
            pass
        if self._clip is None:
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
            self._tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            self._model = AutoModel.from_pretrained(name, local_files_only=True).to(self.device)
        self._model.eval()

    def encode(self, phrases: list[str]):
        np, torch = self.np, self.torch
        chunks = []
        for start in range(0, len(phrases), self.batch_size):
            batch = phrases[start:start + self.batch_size]
            with torch.no_grad():
                if self._clip is not None:
                    tokens = self._clip.tokenize(batch, truncate=True).to(self.device)
                    emb = self._model.encode_text(tokens)
                else:
                    inputs = self._tokenizer(
                        batch, padding="max_length", truncation=True, return_tensors="pt",
                    ).to(self.device)
                    emb = self._model.get_text_features(**inputs)
            chunks.append(emb.float().cpu().numpy())
        if not chunks:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalise_rows(np, np.concatenate(chunks))


def encode_cached(np, encoder, phrases: list[str], cache_dir: Path | None):
    """Encode *phrases*, reusing an earlier result for the same list and backend."""
    if cache_dir is None or not encoder.cacheable or not phrases:
        return encoder.encode(phrases)
    digest = hashlib.sha1("\n".join([encoder.backend_id, *phrases]).encode("utf-8")).hexdigest()
    path = cache_dir / f"{digest}.npy"
    if path.is_file():
        try:
            mat = np.load(path)
            if mat.shape[0] == len(phrases):
                return mat
        except (OSError, ValueError):
            pass
    mat = encoder.encode(phrases)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, mat)
    return mat


def apply_semantic_tier(
    np,
    encoder,
    known: KnownConcepts,
    pending: list[Candidate],
    *,
    max_words: int,
    top_k: int,
    threshold: float | None,
    cache_dir: Path | None,
    allowed_targets: set[str],
) -> int:
    """Annotate *pending* with neighbours and targets; return how many it claimed."""
    pool = known.neighbour_pool(max_words)
    if not pool or not pending:
        return 0
    by_file: dict[str, list[str]] = defaultdict(list)
    for concept, label in pool:
        by_file[label].append(concept)
    labels, concepts, blocks = [], [], []
    for label in sorted(by_file):
        # Encoded per file so that editing one list re-encodes only that one.
        blocks.append(encode_cached(np, encoder, by_file[label], cache_dir))
        labels.extend([label] * len(by_file[label]))
        concepts.extend(by_file[label])
    pool_mat = np.concatenate(blocks)
    cand_mat = encoder.encode([c.surface for c in pending])

    k = min(top_k, len(concepts))
    claimed = 0
    for start in range(0, len(pending), 1024):
        sims = cand_mat[start:start + 1024] @ pool_mat.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for row, cand in enumerate(pending[start:start + 1024]):
            if not cand_mat[start + row].any():
                continue  # no vector for any of its words
            idx = sorted(top[row], key=lambda j: -sims[row, j])
            cand.neighbours = [[concepts[j], labels[j], round(float(sims[row, j]), 4)] for j in idx]
            votes: Counter = Counter()
            for j in idx:
                if not allowed_targets or labels[j] in allowed_targets:
                    votes[labels[j]] += max(0.0, float(sims[row, j]))
            if votes:
                cand.target, cand.target_via = votes.most_common(1)[0][0], "semantic"
            best = idx[0]
            best_sim = float(sims[row, best])
            if threshold is not None and best_sim >= threshold:
                cand.tier, cand.score = "semantic", best_sim
                cand.match, cand.match_file = concepts[best], labels[best]
                claimed += 1
    return claimed


# ---------------------------------------------------------------------------
# Blacklist
# ---------------------------------------------------------------------------

def load_blacklist_checker():
    """A ``concept -> bool`` check against the app's blacklist, or None."""
    os.chdir(PROJECT_ROOT)
    try:
        from teikr.prompts import blacklist_state  # noqa: PLC0415
        from teikr.prompts.blacklist import Blacklist  # noqa: PLC0415
        blacklist_state.set_blacklist()
        if Blacklist.is_empty():
            return None
    except Exception as e:
        print(f"Blacklist could not be loaded: {e}", file=sys.stderr)
        return None
    return lambda concept: Blacklist.get_violation_item(concept) is not None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _rank(c: Candidate):
    return (-c.count, -c.n_words, c.key)


def print_report(cands: list[Candidate], stats: dict, max_list: int, show_blacklisted: bool) -> None:
    print(f"Documents        : {stats['documents']} ({stats['words']:,} words)")
    scores = ", ".join(f"{code} {value:.2f}" for code, value in
                       sorted(stats["language_scores"].items(), key=lambda kv: -kv[1]))
    print(f"Language         : {stats['language']} ({stats['language_method']}"
          f"{'; ' + scores if scores else ''})")
    print(f"Concept set      : {stats['concept_set']}")
    print(f"Known concepts   : {stats['known_entries']:,} entries in {stats['known_files']} files"
          f"{'' if stats['dictionary'] else ' (dictionary excluded)'}")
    print(f"Candidates       : {len(cands):,} with count >= {stats['min_count']}"
          f" ({stats['subsumed']:,} subsumed by longer phrases)")
    if stats.get("semantic_backend"):
        print(f"Semantic backend : {stats['semantic_backend']}")
    print(f"Elapsed          : {stats['elapsed']:.1f}s")
    print()
    tiers = Counter(c.tier for c in cands)
    for tier in ("exact", "variant", "fuzzy", "semantic", "blacklisted", "new"):
        if tiers.get(tier) or tier in ("exact", "new"):
            print(f"  {tier:<12}{tiers.get(tier, 0):>8,}")
    print()

    new = sorted((c for c in cands if c.tier == "new"), key=_rank)
    shown = new if max_list <= 0 else new[:max_list]
    print(f"=== New concepts ({len(shown)} of {len(new)}) ===")
    width = max([len(c.surface) for c in shown] + [7])
    for c in shown:
        target = c.target or "-"
        via = f" ({c.target_via})" if c.target_via else ""
        near = ""
        if c.neighbours:
            near = f"  ~ {c.neighbours[0][0]} {c.neighbours[0][2]:.2f}"
        print(f"  {c.count:>5}  {c.surface:<{width}}  -> {target}{via}{near}")
    print()

    near_dups = sorted((c for c in cands if c.tier in ("fuzzy", "semantic")), key=_rank)
    if near_dups:
        shown = near_dups if max_list <= 0 else near_dups[:max_list]
        print(f"=== Near-duplicates of known concepts ({len(shown)} of {len(near_dups)}) ===")
        width = max(len(c.surface) for c in shown)
        for c in shown:
            print(f"  {c.count:>5}  {c.surface:<{width}}  = {c.match}  [{c.match_file}]"
                  f"  {c.tier} {c.score:.2f}")
        print()

    blacklisted = sorted((c for c in cands if c.tier == "blacklisted"), key=_rank)
    if blacklisted:
        if show_blacklisted:
            print(f"=== Withheld by the blacklist ({len(blacklisted)}) ===")
            for c in blacklisted:
                print(f"  {c.count:>5}  {c.surface}")
        else:
            print(f"{len(blacklisted)} candidates withheld by the blacklist (--show-blacklisted lists them).")
        print()


def export_new(cands: list[Candidate], export_dir: Path) -> dict[str, int]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for c in cands:
        if c.tier == "new":
            buckets[c.target or "_unassigned.txt"].add(c.surface)
    written = {}
    for label, concepts in sorted(buckets.items()):
        out = export_dir / label
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = sorted(concepts, key=lambda s: (s.casefold(), s))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[label] = len(lines)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract concepts from documents that the concept files do not have yet.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Documents or directories to read.")
    g = p.add_argument_group("extraction")
    g.add_argument("--max-ngram", type=int, default=3, help="Longest phrase in words (default: 3).")
    g.add_argument("--min-count", type=int, default=2,
                   help="Occurrences a phrase needs across all inputs (default: 2).")
    g.add_argument("--min-chars", type=int, default=3, help="Shortest phrase in characters (default: 3).")
    g.add_argument("--stopwords", type=Path, action="append", default=[],
                   help="Extra stopword file, one word per line (repeatable).")
    g.add_argument("--no-subsume", action="store_true",
                   help="Keep phrases that only ever occur inside one longer phrase.")
    g = p.add_argument_group("known concepts")
    g.add_argument("--concepts-dir", type=Path,
                   help="Concept set to compare against. Default: the set in the repo root whose "
                        "concept_set.json names the documents' language.")
    g.add_argument("--language",
                   help="Language of the documents, e.g. en or de, instead of detecting it.")
    g.add_argument("--concept-set", type=Path, action="append", default=[],
                   help="Another concept set folder to choose from by language (repeatable).")
    g.add_argument("--include-dictionary", action="store_true",
                   help="Count dictionary.txt as known, so single common words are not new.")
    g.add_argument("--known", type=Path, action="append", default=[],
                   help="Extra list of known concepts, e.g. an earlier export (repeatable).")
    g.add_argument("--target-file", action="append", default=[],
                   help="Restrict suggested targets to these concept files, named as in the "
                        "report, e.g. object.txt or _/style_anime.txt (repeatable).")
    g = p.add_argument_group("matching")
    g.add_argument("--fuzzy-threshold", type=float, default=0.88,
                   help="difflib ratio for a spelling variant (default: 0.88).")
    g.add_argument("--no-fuzzy", action="store_true")
    g.add_argument("--vectors", type=Path,
                   help="Word vector file (GloVe / word2vec / fastText text, optionally .gz).")
    g.add_argument("--clip", metavar="MODEL",
                   help="CLIP model name or .pt file, or a cached Hugging Face CLIP/SigLIP id.")
    g.add_argument("--device", help="Torch device for --clip (default: cuda when available).")
    g.add_argument("--batch-size", type=int, default=256, help="Batch size for --clip (default: 256).")
    g.add_argument("--semantic-threshold", type=float,
                   help="Cosine similarity at which a semantic neighbour counts as the same "
                        "concept. Unset, vectors only suggest targets.")
    g.add_argument("--top-k", type=int, default=5, help="Neighbours voting on a target (default: 5).")
    g.add_argument("--vector-cache", type=Path,
                   default=Path.home() / ".cache" / "sd_runner" / "concept_vectors",
                   help="Where --clip embeddings of the concept files are cached.")
    g.add_argument("--no-vector-cache", action="store_true")
    g = p.add_argument_group("output")
    g.add_argument("--skip-blacklist", action="store_true", help="Do not filter against the blacklist.")
    g.add_argument("--show-blacklisted", action="store_true")
    g.add_argument("--max-list", type=int, default=200, help="Rows per report section; 0 for all (default: 200).")
    g.add_argument("--json", type=Path, help="Write every candidate with its tier and match as JSON.")
    g.add_argument("--export-dir", type=Path,
                   help="Write new concepts into one file per suggested target under this directory.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    if args.vectors and args.clip:
        print("Error: use either --vectors or --clip, not both.", file=sys.stderr)
        return 2
    if args.semantic_threshold is not None and not (args.vectors or args.clip):
        print("Warning: --semantic-threshold has no effect without --vectors or --clip.", file=sys.stderr)
    if args.max_ngram < 1 or args.min_count < 1:
        print("Error: --max-ngram and --min-count must be at least 1.", file=sys.stderr)
        return 2
    language_arg = None
    if args.language is not None:
        language_arg = normalize_language_code(args.language)
        if language_arg is None:
            if args.language.strip().lower().replace("_", "-").split("-", 1)[0] != "en":
                print(f"Error: not a language code: {args.language!r}", file=sys.stderr)
                return 2
            language_arg = "en"
    export_dir = args.export_dir.resolve() if args.export_dir else None
    json_path = args.json.resolve() if args.json else None
    inputs = [p.resolve() for p in args.inputs]
    extra_stopwords: set[str] = set()
    for path in args.stopwords:
        extra_stopwords.update(
            w.strip().casefold() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())

    # --- Documents ---------------------------------------------------------
    documents: list[tuple[str, str]] = []
    for path in iter_document_paths(inputs):
        try:
            documents.append((str(path), read_document(path)))
        except Exception as e:
            print(f"Skipped {path}: {e}", file=sys.stderr)
    if not documents:
        print("Error: no readable documents.", file=sys.stderr)
        return 1

    # --- Language and concept set -------------------------------------------
    sets = discover_concept_sets(PROJECT_ROOT, [p.resolve() for p in args.concept_set])
    if args.concepts_dir is not None:
        concept_set = read_concept_set(args.concepts_dir.resolve())
        guess = LanguageGuess(language_arg or concept_set.language,
                              "--language" if language_arg else "manifest")
    else:
        if language_arg:
            guess = LanguageGuess(language_arg, "--language")
        elif sets:
            guess = detect_language(documents, frozenset(s.language for s in sets))
        else:
            guess = LanguageGuess(None, "no concept sets")
        if guess.language is None:
            fallback = next((s for s in sets if s.language == "en"), sets[0] if sets else None)
            if fallback is None:
                print(f"Error: no folder with a {MANIFEST_FILENAME} in {PROJECT_ROOT}; "
                      "pass --concepts-dir.", file=sys.stderr)
                return 1
            print(f"Warning: the documents' language could not be told ({guess.method}); "
                  f"using {fallback.language}. Pass --language to choose.", file=sys.stderr)
            guess.language = fallback.language
        matching = [s for s in sets if s.language == guess.language]
        if not matching:
            available = ", ".join(sorted({s.language for s in sets})) or "none"
            print(f"Error: no concept set for language {guess.language!r} (available: {available}); "
                  "pass --concepts-dir.", file=sys.stderr)
            return 1
        concept_set = matching[0]
        if len(matching) > 1:
            print(f"Warning: several concept sets are in {guess.language}; using {concept_set.path}. "
                  "Pass --concepts-dir to choose.", file=sys.stderr)
    language = guess.language
    concepts_dir = concept_set.path
    if not concepts_dir.is_dir():
        print(f"Error: concepts directory not found: {concepts_dir}", file=sys.stderr)
        return 1
    others = sorted(label for label, code in guess.per_document.items() if code and code != language)
    if others:
        print(f"Warning: {len(others)} documents look like another language than {language}: "
              + ", ".join(others[:5]) + (" ..." if len(others) > 5 else ""), file=sys.stderr)
    protected = {concepts_dir} | {s.path for s in sets}
    if export_dir is not None and any(export_dir == d or export_dir in d.parents for d in protected):
        print("Error: --export-dir must not be a concept set folder or contain one.", file=sys.stderr)
        return 2
    if args.clip and language != "en":
        print("Warning: CLIP text encoders are trained on English text; similarities in "
              f"{language} are unreliable.", file=sys.stderr)

    base_stopwords = STOPWORDS_BY_LANGUAGE.get(language)
    if base_stopwords is None:
        base_stopwords = frozenset()
        print(f"Warning: no built-in stopwords for {language}; pass --stopwords for sensible "
              "phrase boundaries.", file=sys.stderr)
    stopwords = frozenset(base_stopwords | extra_stopwords)

    cands_by_key, total_words = extract_candidates(
        documents, max_ngram=args.max_ngram, min_chars=args.min_chars, stopwords=stopwords,
    )
    subsumed = 0 if args.no_subsume else drop_subsumed(cands_by_key, args.min_count)
    cands = [c for c in cands_by_key.values() if c.count >= args.min_count]

    # --- Known concepts ----------------------------------------------------
    files = discover_concept_files(concepts_dir, args.include_dictionary)
    known = KnownConcepts(files, [p.resolve() for p in args.known], args.max_ngram, language)
    allowed_targets = set(args.target_file)

    # --- Tiers: exact, variant, fuzzy --------------------------------------
    pending: list[Candidate] = []
    for c in cands:
        idx = known.exact.get(exact_key(c.surface))
        tier = "exact"
        if idx is None:
            idx = known.variant.get(variant_key(c.surface, language))
            tier = "variant"
        if idx is not None:
            c.tier, c.score = tier, 1.0
            c.match, c.match_file = known.entries[idx]
        else:
            pending.append(c)

    if not args.no_fuzzy and pending:
        index = known.fuzzy()
        still = []
        for c in pending:
            vkey = variant_key(c.surface, language)
            ratio, i = (0.0, None) if len(vkey) < 5 else index.best(vkey, args.fuzzy_threshold)
            if i is not None:
                c.tier, c.score = "fuzzy", ratio
                c.match, c.match_file = known.entries[known.variant[index.keys[i]]]
            else:
                still.append(c)
        pending = still

    # --- Target by head word ------------------------------------------------
    for c in pending:
        target = known.suggest_by_head(variant_key(c.surface, language), allowed_targets)
        if target:
            c.target, c.target_via = target, "head"

    # --- Semantic tier (optional) --------------------------------------------
    semantic_backend = None
    if (args.vectors or args.clip) and pending:
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            print("Error: the semantic tier needs numpy.", file=sys.stderr)
            return 1
        if args.vectors:
            phrases = [c.surface for c in pending] + [p for p, _l in known.neighbour_pool(args.max_ngram)]
            encoder = WordVectorEncoder(np, args.vectors.resolve(), WordVectorEncoder.vocabulary(phrases), stopwords)
        else:
            encoder = ClipTextEncoder(np, args.clip, args.device, args.batch_size)
        semantic_backend = encoder.backend_id
        cache_dir = None if args.no_vector_cache else args.vector_cache
        apply_semantic_tier(
            np, encoder, known, pending,
            max_words=args.max_ngram, top_k=args.top_k,
            threshold=args.semantic_threshold, cache_dir=cache_dir,
            allowed_targets=allowed_targets,
        )
        pending = [c for c in pending if c.tier == "new"]

    # --- Blacklist ------------------------------------------------------------
    if not args.skip_blacklist:
        is_blacklisted = load_blacklist_checker()
        if is_blacklisted is None:
            print("Blacklist not loaded; nothing was filtered. Pass --skip-blacklist "
                  "to run without it.", file=sys.stderr)
            if export_dir is not None:
                print("Error: refusing to export unfiltered concepts.", file=sys.stderr)
                return 1
        else:
            for c in pending:
                if is_blacklisted(c.surface):
                    c.tier = "blacklisted"

    # --- Output ---------------------------------------------------------------
    stats = {
        "documents": len(documents),
        "words": total_words,
        "language": language,
        "language_method": guess.method,
        "language_scores": {code: round(value, 4) for code, value in guess.scores.items()},
        "concept_set": str(concepts_dir),
        "known_entries": len(known.entries),
        "known_files": known.file_count,
        "dictionary": args.include_dictionary,
        "min_count": args.min_count,
        "subsumed": subsumed,
        "semantic_backend": semantic_backend,
        "elapsed": time.perf_counter() - started,
    }
    print_report(cands, stats, args.max_list, args.show_blacklisted)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stats": stats,
            "candidates": [c.to_dict() for c in sorted(cands, key=_rank)],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {json_path}")

    if export_dir is not None:
        written = export_new(cands, export_dir)
        for label, count in written.items():
            print(f"  {count:>5}  {export_dir / label}")
        print(f"Exported {sum(written.values())} new concepts into {len(written)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
