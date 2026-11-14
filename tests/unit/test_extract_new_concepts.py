"""Tests for scripts/extract_new_concepts.py."""

import importlib.util
import json
import os
import sys
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def tool():
    # Registered before exec_module: with postponed annotations, @dataclass
    # looks its class's module up in sys.modules while the module is running.
    name = "extract_new_concepts"
    path = os.path.join(REPO_ROOT, "scripts", "extract_new_concepts.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def no_blacklist(tool, monkeypatch):
    """A blacklist that withholds nothing, without loading the app's."""
    monkeypatch.setattr(tool, "load_blacklist_checker", lambda: (lambda concept: False))


def write_lines(path: Path, *lines: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_concepts_dir(root: Path) -> Path:
    concepts = root / "concepts"
    write_lines(concepts / "object.txt", "castle", "glass lantern", "lantern")
    write_lines(concepts / "media_features.txt", "watercolour")
    write_lines(concepts / "dictionary.txt", "paper")
    write_lines(concepts / "nsfw_concepts.txt", "forbidden thing")
    write_lines(concepts / "_" / "style_general.txt", "art nouveau")
    return concepts


def candidates(tool, text, *, max_ngram=3, min_chars=3, stopwords=None):
    cands, _words = tool.extract_candidates(
        [("doc", text)], max_ngram=max_ngram, min_chars=min_chars,
        stopwords=tool.STOPWORDS if stopwords is None else stopwords,
    )
    return cands


def make_candidate(tool, surface, count=1):
    return tool.Candidate(
        key=surface.casefold(), n_words=len(surface.split()), count=count,
        surfaces=Counter({surface: count}),
    )


# ---------------------------------------------------------------------------
# Reading documents
# ---------------------------------------------------------------------------

class TestReadDocument:
    def test_html_drops_scripts_and_styles(self, tool, tmp_path):
        page = write_lines(tmp_path / "page.html",
                           "<html><head><title>Title</title><style>p {}</style></head>",
                           "<body><script>var x;</script><p>visible text</p></body></html>")
        text = tool.read_document(page)
        assert "visible text" in text
        assert "var x" not in text and "p {}" not in text

    def test_an_unclosed_head_does_not_hide_the_body(self, tool, tmp_path):
        page = write_lines(tmp_path / "page.html", "<head><title>T</title><p>body text</p>")
        assert "body text" in tool.read_document(page)

    def test_html_blocks_become_phrase_boundaries(self, tool, tmp_path):
        page = write_lines(tmp_path / "page.html", "<li>red</li><li>apple</li>")
        cands = candidates(tool, tool.read_document(page), min_chars=1)
        assert "red apple" not in cands

    def test_epub_reads_every_content_document(self, tool, tmp_path):
        book = tmp_path / "book.epub"
        with zipfile.ZipFile(book, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("OEBPS/ch1.xhtml", "<html><body><p>first chapter</p></body></html>")
            zf.writestr("OEBPS/ch2.html", "<html><body><p>second chapter</p></body></html>")
        text = tool.read_document(book)
        assert "first chapter" in text and "second chapter" in text

    def test_binary_files_are_refused(self, tool, tmp_path):
        blob = tmp_path / "blob.dat"
        blob.write_bytes(b"\x00\x01\x02binary")
        with pytest.raises(ValueError):
            tool.read_document(blob)

    def test_non_utf8_text_falls_back_to_cp1252(self, tool, tmp_path):
        doc = tmp_path / "old.txt"
        doc.write_bytes("café crème".encode("cp1252"))
        assert tool.read_document(doc) == "café crème"

    def test_directories_are_searched_without_hidden_entries(self, tool, tmp_path):
        docs = tmp_path / "docs"
        write_lines(docs / "a.txt", "x")
        write_lines(docs / "sub" / "b.txt", "x")
        write_lines(docs / ".hidden" / "c.txt", "x")
        write_lines(docs / ".d.txt", "x")
        found = {p.name for p in tool.iter_document_paths([docs])}
        assert found == {"a.txt", "b.txt"}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestKeys:
    @pytest.mark.parametrize("a, b", [
        ("Glass Lanterns", "glass lantern"),
        ("facing_away", "facing away"),
        ("well-known", "well known"),
        ("Café", "cafe"),
        ("berries", "berry"),
        ("boxes", "box"),
        ("glasses", "glass"),
        ("dragon's hoard", "dragon hoard"),
        ("dragon’s hoard", "dragon hoard"),
    ])
    def test_variants_share_a_key(self, tool, a, b):
        assert tool.variant_key(a) == tool.variant_key(b)

    @pytest.mark.parametrize("word", ["glass", "cactus", "axis", "bus"])
    def test_words_ending_in_s_that_are_not_plurals_keep_it(self, tool, word):
        assert tool.variant_key(word) == word

    def test_exact_key_folds_case_and_whitespace_only(self, tool):
        assert tool.exact_key("  Glass \t Lantern ") == "glass lantern"
        assert tool.exact_key("glass-lantern") != tool.exact_key("glass lantern")


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_phrases_do_not_cross_punctuation(self, tool):
        cands = candidates(tool, "red apple, green pear")
        assert "red apple" in cands and "green pear" in cands
        assert "apple green" not in cands

    @pytest.mark.parametrize("separator", [" 3 ", " -- ", "--", " - ", " — ", "/", " (", ";"])
    def test_separators_break_adjacency(self, tool, separator):
        cands = candidates(tool, f"red{separator}apples")
        assert "red apples" not in cands

    def test_a_line_wrap_does_not_break_a_phrase(self, tool):
        assert "paper lantern" in candidates(tool, "a paper\nlantern")

    def test_a_blank_line_breaks_a_phrase(self, tool):
        assert "paper lantern" not in candidates(tool, "a paper\n\nlantern")

    def test_hyphenated_words_stay_one_token(self, tool):
        cands = candidates(tool, "a well-known knight")
        assert "well-known knight" in cands
        assert cands["well-known knight"].n_words == 2

    def test_stopwords_may_not_start_or_end_a_phrase(self, tool):
        cands = candidates(tool, "the cup of tea")
        assert "cup of tea" in cands
        assert not any(k.startswith("the ") or k.endswith(" of") for k in cands)

    def test_max_ngram_bounds_phrase_length(self, tool):
        cands = candidates(tool, "ancient stone bridge arch", max_ngram=2)
        assert max(c.n_words for c in cands.values()) == 2

    def test_min_chars_drops_short_phrases(self, tool):
        assert "ox" not in candidates(tool, "an ox", min_chars=3)
        assert "ox" in candidates(tool, "an ox", min_chars=2)

    def test_counts_and_documents_accumulate(self, tool):
        cands, words = tool.extract_candidates(
            [("a", "paper lantern."), ("b", "Paper lantern. paper lantern.")],
            max_ngram=2, min_chars=3, stopwords=tool.STOPWORDS,
        )
        c = cands["paper lantern"]
        assert c.count == 3
        assert c.documents == {"a", "b"}
        assert words == 6

    def test_the_lower_case_spelling_is_preferred(self, tool):
        cands = candidates(tool, "Paper Lantern. Paper Lantern. paper lantern.")
        assert cands["paper lantern"].surface == "paper lantern"

    def test_a_proper_noun_keeps_its_capitals(self, tool):
        assert candidates(tool, "Paris. Paris.")["paris"].surface == "Paris"

    def test_extra_stopwords_apply(self, tool):
        cands = candidates(tool, "der Hund", stopwords=tool.STOPWORDS | {"der"})
        assert "der hund" not in cands and "hund" in cands


class TestSubsumption:
    def test_a_phrase_only_seen_inside_a_longer_one_is_dropped(self, tool):
        cands = candidates(tool, "golden retriever puppy. golden retriever puppy.")
        dropped = tool.drop_subsumed(cands, min_count=2)
        assert "golden retriever puppy" in cands
        assert "golden retriever" not in cands and "retriever puppy" not in cands
        assert dropped >= 2

    def test_a_phrase_also_seen_alone_is_kept(self, tool):
        cands = candidates(tool, "golden retriever puppy. golden retriever puppy. golden retriever.")
        tool.drop_subsumed(cands, min_count=2)
        assert "golden retriever" in cands

    def test_longer_phrases_below_min_count_subsume_nothing(self, tool):
        cands = candidates(tool, "golden retriever puppy.")
        tool.drop_subsumed(cands, min_count=2)
        assert "golden retriever" in cands


# ---------------------------------------------------------------------------
# Known concepts and fuzzy matching
# ---------------------------------------------------------------------------

class TestKnownConcepts:
    def test_dictionary_is_only_read_on_request(self, tool, tmp_path):
        concepts = make_concepts_dir(tmp_path)
        without = {label for label, _p in tool.discover_concept_files(concepts, False)}
        with_dict = {label for label, _p in tool.discover_concept_files(concepts, True)}
        assert "dictionary.txt" not in without
        assert with_dict - without == {"dictionary.txt"}
        assert "_/style_general.txt" in without

    @pytest.mark.parametrize("label, expected", [
        ("object.txt", True),
        ("_/style_general.txt", True),
        ("dictionary.txt", False),
        ("negatives.txt", False),
        ("nsfw_concepts.txt", False),
        ("nsfl_actions.txt", False),
    ])
    def test_target_files(self, tool, label, expected):
        assert tool.is_target_file(label) is expected

    def test_lines_are_parsed_like_the_concepts_loader(self, tool, tmp_path):
        f = write_lines(tmp_path / "object.txt", "lantern  # a comment", "# only a comment", "", "  rope  ")
        known = tool.KnownConcepts([("object.txt", f)], [], max_words=3)
        assert [c for c, _l in known.entries] == ["lantern", "rope"]

    def test_template_lines_are_matched_only_by_their_raw_text(self, tool, tmp_path):
        f = write_lines(tmp_path / "object.txt", "[[red,blue]] lantern")
        known = tool.KnownConcepts([("object.txt", f)], [], max_words=3)
        assert tool.exact_key("[[red,blue]] lantern") in known.exact
        assert not known.variant
        assert not known.head_files

    def test_head_words_come_only_from_target_files(self, tool, tmp_path):
        files = [
            ("object.txt", write_lines(tmp_path / "object.txt", "paper lantern")),
            ("nsfw_concepts.txt", write_lines(tmp_path / "nsfw_concepts.txt", "red rope")),
        ]
        known = tool.KnownConcepts(files, [], max_words=3)
        assert known.suggest_by_head("stone lantern", set()) == "object.txt"
        assert known.suggest_by_head("silk rope", set()) is None

    def test_head_suggestion_honours_allowed_targets(self, tool, tmp_path):
        files = [
            ("object.txt", write_lines(tmp_path / "object.txt", "lantern", "paper lantern")),
            ("lighting.txt", write_lines(tmp_path / "lighting.txt", "lantern light", "lantern")),
        ]
        known = tool.KnownConcepts(files, [], max_words=3)
        assert known.suggest_by_head("stone lantern", set()) == "object.txt"
        assert known.suggest_by_head("stone lantern", {"lighting.txt"}) == "lighting.txt"
        assert known.suggest_by_head("stone lantern", {"colors.txt"}) is None

    def test_extra_known_lists_count_but_are_never_targets(self, tool, tmp_path):
        extra = write_lines(tmp_path / "earlier_export.txt", "paper lantern")
        known = tool.KnownConcepts([], [extra], max_words=3)
        assert tool.exact_key("paper lantern") in known.exact
        assert known.suggest_by_head("stone lantern", set()) is None
        assert known.neighbour_pool(3) == []


class TestFuzzyIndex:
    KEYS = [
        "watercolour", "watercolor painting", "lantern", "paper lantern",
        "castle", "castles in the air", "dragon", "wagon", "dragonfly",
        "colour", "color", "art nouveau", "art deco", "stone bridge",
    ]

    def test_finds_a_spelling_variant(self, tool):
        index = tool.FuzzyIndex(self.KEYS)
        ratio, i = index.best("watercolor", 0.88)
        assert index.keys[i] == "watercolour"
        assert ratio >= 0.88

    def test_rejects_words_that_only_look_alike(self, tool):
        index = tool.FuzzyIndex(self.KEYS)
        assert index.best("drogan", 0.88)[1] is None
        assert index.best("wagons", 0.88)[1] is not None
        assert index.best("flagon", 0.88)[1] is None

    def test_empty_query(self, tool):
        assert tool.FuzzyIndex(self.KEYS).best("", 0.88) == (0.0, None)

    @pytest.mark.parametrize("query", [
        "watercolor", "lanterns", "paper lanterns", "castel", "art noveau",
        "dragonfli", "stone bridges", "colr", "unrelated phrase",
    ])
    @pytest.mark.parametrize("threshold", [0.7, 0.88])
    def test_prefix_filtering_matches_a_full_scan(self, tool, query, threshold):
        # The index probes only the query's rarest trigrams; it must find the
        # same best ratio as checking every key under the same bounds.
        index = tool.FuzzyIndex(self.KEYS)
        q = index._trigrams(query)
        expected = 0.0
        for key in self.KEYS:
            r = index._trigrams(key)
            inter = len(q & r)
            if inter / (len(q) + len(r) - inter) < index.floor:
                continue
            ratio = SequenceMatcher(None, query, key, autojunk=False).ratio()
            if ratio >= threshold:
                expected = max(expected, ratio)
        assert index.best(query, threshold)[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Semantic tier
# ---------------------------------------------------------------------------

class FakeEncoder:
    backend_id = "fake"
    cacheable = False

    def __init__(self, np, vectors):
        self.np = np
        self.vectors = vectors
        self.calls = 0

    def encode(self, phrases):
        self.calls += 1
        rows = [self.vectors.get(p, [0.0, 0.0]) for p in phrases]
        return self.np.array(rows, dtype=self.np.float32).reshape(len(phrases), 2)


@pytest.fixture
def np():
    return pytest.importorskip("numpy")


class TestWordVectorEncoder:
    def test_reads_only_the_vocabulary_and_averages_content_words(self, tool, np, tmp_path):
        vec = write_lines(tmp_path / "glove.txt", "cat 1 0", "dog 0 1", "the 5 5", "unused 3 3")
        enc = tool.WordVectorEncoder(np, vec, {"cat", "dog", "the"}, tool.STOPWORDS)
        assert set(enc.vectors) == {"cat", "dog", "the"}
        out = enc.encode(["cat", "the dog", "cat dog", "zebra"])
        assert out[0] == pytest.approx([1, 0])
        assert out[1] == pytest.approx([0, 1])
        assert out[2] == pytest.approx([2 ** -0.5, 2 ** -0.5])
        assert not out[3].any()

    def test_skips_a_word2vec_header_and_reads_gzip(self, tool, np, tmp_path):
        import gzip
        vec = tmp_path / "vectors.vec.gz"
        with gzip.open(vec, "wt", encoding="utf-8") as f:
            f.write("2 2\ncat 1 0\nnew_york 0 1\n")
        enc = tool.WordVectorEncoder(np, vec, {"cat", "new_york", "new", "york"}, tool.STOPWORDS)
        assert enc.dim == 2
        assert enc.encode(["New York"])[0] == pytest.approx([0, 1])

    def test_no_usable_vectors_is_an_error(self, tool, np, tmp_path):
        vec = write_lines(tmp_path / "glove.txt", "cat 1 0")
        with pytest.raises(RuntimeError):
            tool.WordVectorEncoder(np, vec, {"dog"}, tool.STOPWORDS)


class TestEncodeCached:
    def test_reuses_the_result_for_the_same_phrases(self, tool, np, tmp_path):
        enc = FakeEncoder(np, {"cat": [1.0, 0.0]})
        enc.cacheable = True
        cache_dir = tmp_path / "vectors"
        first = tool.encode_cached(np, enc, ["cat"], cache_dir)
        second = tool.encode_cached(np, enc, ["cat"], cache_dir)
        assert enc.calls == 1
        assert (first == second).all()
        tool.encode_cached(np, enc, ["cat", "dog"], cache_dir)
        assert enc.calls == 2

    def test_uncacheable_encoders_always_encode(self, tool, np, tmp_path):
        enc = FakeEncoder(np, {})
        cache_dir = tmp_path / "vectors"
        tool.encode_cached(np, enc, ["cat"], cache_dir)
        tool.encode_cached(np, enc, ["cat"], cache_dir)
        assert enc.calls == 2
        assert not cache_dir.exists()


class TestSemanticTier:
    VECTORS = {
        "cat": [1.0, 0.0],
        "car": [0.0, 1.0],
        "kitten": [0.99, 0.14],
    }

    @pytest.fixture
    def known(self, tool, tmp_path):
        files = [
            ("animals.txt", write_lines(tmp_path / "animals.txt", "cat")),
            ("object.txt", write_lines(tmp_path / "object.txt", "car")),
        ]
        return tool.KnownConcepts(files, [], max_words=3)

    def run(self, tool, np, known, pending, threshold=None, top_k=1, allowed=frozenset()):
        return tool.apply_semantic_tier(
            np, FakeEncoder(np, self.VECTORS), known, pending,
            max_words=3, top_k=top_k, threshold=threshold, cache_dir=None,
            allowed_targets=set(allowed),
        )

    def test_without_a_threshold_it_only_suggests(self, tool, np, known):
        kitten = make_candidate(tool, "kitten")
        assert self.run(tool, np, known, [kitten]) == 0
        assert kitten.tier == "new"
        assert (kitten.target, kitten.target_via) == ("animals.txt", "semantic")
        assert kitten.neighbours[0][:2] == ["cat", "animals.txt"]

    def test_a_threshold_claims_close_neighbours(self, tool, np, known):
        kitten = make_candidate(tool, "kitten")
        assert self.run(tool, np, known, [kitten], threshold=0.95) == 1
        assert (kitten.tier, kitten.match, kitten.match_file) == ("semantic", "cat", "animals.txt")

    def test_a_threshold_above_the_similarity_claims_nothing(self, tool, np, known):
        kitten = make_candidate(tool, "kitten")
        assert self.run(tool, np, known, [kitten], threshold=0.999) == 0
        assert kitten.tier == "new"

    def test_a_phrase_without_a_vector_is_left_alone(self, tool, np, known):
        blorp = make_candidate(tool, "blorp")
        blorp.target, blorp.target_via = "object.txt", "head"
        self.run(tool, np, known, [blorp], threshold=0.0)
        assert (blorp.tier, blorp.target, blorp.target_via) == ("new", "object.txt", "head")
        assert blorp.neighbours == []

    def test_allowed_targets_restrict_the_vote(self, tool, np, known):
        kitten = make_candidate(tool, "kitten")
        self.run(tool, np, known, [kitten], top_k=2, allowed={"object.txt"})
        assert kitten.target == "object.txt"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

DOCUMENT = "castle, paper lantern, Watercolor, glass lanterns, paper.\n"


def run_main(tool, tmp_path, *extra):
    concepts = make_concepts_dir(tmp_path)
    doc = write_lines(tmp_path / "doc.txt", DOCUMENT, "", DOCUMENT)
    out = tmp_path / "out.json"
    code = tool.main([str(doc), "--concepts-dir", str(concepts), "--json", str(out), *extra])
    result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return code, result


def by_concept(result):
    return {c["concept"]: c for c in result["candidates"]}


@pytest.mark.usefixtures("no_blacklist")
class TestMain:
    def test_each_tier_claims_its_candidates(self, tool, tmp_path):
        code, result = run_main(tool, tmp_path)
        assert code == 0
        found = by_concept(result)
        assert found["castle"]["tier"] == "exact"
        assert found["glass lanterns"]["tier"] == "variant"
        assert found["glass lanterns"]["match"] == "glass lantern"
        assert found["Watercolor"]["tier"] == "fuzzy"
        assert found["Watercolor"]["match_file"] == "media_features.txt"
        assert found["paper lantern"]["tier"] == "new"
        assert (found["paper lantern"]["target"], found["paper lantern"]["target_via"]) == ("object.txt", "head")

    def test_the_dictionary_counts_as_known_on_request(self, tool, tmp_path):
        _code, without = run_main(tool, tmp_path)
        assert by_concept(without)["paper"]["tier"] == "new"
        _code, with_dict = run_main(tool, tmp_path, "--include-dictionary")
        assert by_concept(with_dict)["paper"]["tier"] == "exact"

    def test_no_fuzzy_leaves_spelling_variants_new(self, tool, tmp_path):
        _code, result = run_main(tool, tmp_path, "--no-fuzzy")
        assert by_concept(result)["Watercolor"]["tier"] == "new"

    def test_min_count_filters_rare_phrases(self, tool, tmp_path):
        # "paper" is the most frequent candidate, at four occurrences.
        _code, result = run_main(tool, tmp_path, "--min-count", "5")
        assert result["candidates"] == []

    def test_export_writes_sorted_files_per_target(self, tool, tmp_path):
        export = tmp_path / "export"
        code, _result = run_main(tool, tmp_path, "--export-dir", str(export), "--no-fuzzy")
        assert code == 0
        assert (export / "object.txt").read_text(encoding="utf-8") == "paper lantern\n"
        unassigned = (export / "_unassigned.txt").read_text(encoding="utf-8").splitlines()
        assert unassigned == sorted(unassigned, key=str.casefold)
        assert "Watercolor" in unassigned and "paper" in unassigned
        assert not (export / "castle.txt").exists()

    def test_blacklisted_candidates_are_withheld_from_the_export(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr(tool, "load_blacklist_checker", lambda: (lambda c: c == "paper lantern"))
        export = tmp_path / "export"
        _code, result = run_main(tool, tmp_path, "--export-dir", str(export))
        assert by_concept(result)["paper lantern"]["tier"] == "blacklisted"
        assert not (export / "object.txt").exists()

    def test_export_is_refused_when_the_blacklist_cannot_load(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr(tool, "load_blacklist_checker", lambda: None)
        export = tmp_path / "export"
        code, _result = run_main(tool, tmp_path, "--export-dir", str(export))
        assert code == 1
        assert not export.exists()

    def test_skip_blacklist_never_loads_it(self, tool, tmp_path, monkeypatch):
        def fail():
            raise AssertionError("blacklist loaded")
        monkeypatch.setattr(tool, "load_blacklist_checker", fail)
        code, _result = run_main(tool, tmp_path, "--skip-blacklist", "--export-dir", str(tmp_path / "export"))
        assert code == 0

    @pytest.mark.parametrize("relative", ["concepts", "."])
    def test_export_into_or_above_the_concepts_dir_is_refused(self, tool, tmp_path, relative):
        before = (make_concepts_dir(tmp_path) / "object.txt").read_text(encoding="utf-8")
        code, _result = run_main(tool, tmp_path, "--export-dir", str(tmp_path / relative))
        assert code == 2
        assert (tmp_path / "concepts" / "object.txt").read_text(encoding="utf-8") == before

    def test_vectors_and_clip_are_exclusive(self, tool, tmp_path):
        code, _result = run_main(tool, tmp_path, "--vectors", "x.txt", "--clip", "ViT-B/32")
        assert code == 2

    def test_no_readable_documents_is_an_error(self, tool, tmp_path):
        concepts = make_concepts_dir(tmp_path)
        assert tool.main([str(tmp_path / "missing.txt"), "--concepts-dir", str(concepts)]) == 1

    def test_word_vectors_suggest_targets_end_to_end(self, tool, np, tmp_path):
        vec = write_lines(tmp_path / "glove.txt", "paper 0 1", "lantern 1 0", "castle 0.9 0.1",
                          "watercolour 0 1", "glass 1 0", "art 0 1", "nouveau 0 1")
        _code, result = run_main(tool, tmp_path, "--vectors", str(vec), "--top-k", "1")
        found = by_concept(result)
        assert found["paper lantern"]["target_via"] == "semantic"
        assert found["paper lantern"]["neighbours"]


# ---------------------------------------------------------------------------
# Language and concept set
# ---------------------------------------------------------------------------

GERMAN_TEXT = "Die Laterne und der Ballon. Die Laterne und der Ballon. Eine Laterne ist nicht ein Ballon.\n"
ENGLISH_TEXT = "The lantern and the balloon. The lantern and the balloon. A lantern is not a balloon.\n"


def write_manifest(folder: Path, language) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "concept_set.json").write_text(json.dumps({"language": language}), encoding="utf-8")
    return folder


@pytest.fixture
def no_lingua(tool, monkeypatch):
    """Detection by stopwords, whether or not Lingua is installed."""
    monkeypatch.setattr(tool, "_lingua_scorer", lambda languages: None)


class TestConceptSets:
    @pytest.mark.parametrize("language, expected", [
        ("de", "de"), ("de_DE", "de"), ("de-AT", "de"), ("en", "en"), ("en-GB", "en"), (None, "en"),
    ])
    def test_the_manifest_names_the_language(self, tool, tmp_path, language, expected):
        folder = write_manifest(tmp_path / "set", language)
        assert tool.read_concept_set(folder).language == expected

    @pytest.mark.parametrize("content", [None, "not json", "[\"de\"]"])
    def test_a_missing_or_unreadable_manifest_means_english(self, tool, tmp_path, content):
        folder = tmp_path / "set"
        folder.mkdir()
        if content is not None:
            (folder / "concept_set.json").write_text(content, encoding="utf-8")
        assert tool.read_concept_set(folder).language == "en"

    def test_sets_are_the_folders_with_a_manifest(self, tool, tmp_path):
        root = tmp_path / "repo"
        write_manifest(root / "concepts", "en")
        write_manifest(root / "Konzepte", "de")
        (root / "docs").mkdir()
        write_lines(root / "concept_set.json", "{}")
        found = {(s.path.name, s.language) for s in tool.discover_concept_sets(root, [])}
        assert found == {("concepts", "en"), ("Konzepte", "de")}

    def test_extra_sets_are_added_once(self, tool, tmp_path):
        root = tmp_path / "repo"
        write_manifest(root / "concepts", "en")
        extra = write_manifest(tmp_path / "elsewhere", "fr")
        found = tool.discover_concept_sets(root, [extra, extra, root / "concepts"])
        assert [(s.path.name, s.language) for s in found] == [("concepts", "en"), ("elsewhere", "fr")]


class TestLanguageDependentKeys:
    def test_plurals_are_reduced_only_for_english(self, tool):
        assert tool.variant_key("berries", "de") == "berries"
        assert tool.variant_key("Glas", "de") == "glas"
        assert tool.variant_key("Glass Lanterns", "en") == "glass lantern"

    def test_other_folding_applies_to_every_language(self, tool):
        assert tool.variant_key("Balsa-Holz", "de") == tool.variant_key("balsa holz", "de")
        assert tool.variant_key("Häuser", "de") == "hauser"

    def test_known_concepts_use_the_language(self, tool, tmp_path):
        # The English rule would cut "Glas" to "gla"; "Haus" would not show
        # it, since a final "us" is kept in English too.
        f = write_lines(tmp_path / "object.txt", "Glas")
        assert set(tool.KnownConcepts([("object.txt", f)], [], 3, "de").variant) == {"glas"}
        assert set(tool.KnownConcepts([("object.txt", f)], [], 3, "en").variant) == {"gla"}


@pytest.mark.usefixtures("no_lingua")
class TestDetectLanguageByStopwords:
    LANGUAGES = frozenset({"en", "de"})

    def test_one_set_needs_no_detection(self, tool, monkeypatch):
        monkeypatch.setattr(tool, "_stopword_scorer", lambda languages: pytest.fail("scored"))
        guess = tool.detect_language([("d", GERMAN_TEXT)], frozenset({"en"}))
        assert (guess.language, guess.method) == ("en", "only set")

    @pytest.mark.parametrize("text, expected", [(GERMAN_TEXT, "de"), (ENGLISH_TEXT, "en")])
    def test_detects_the_language(self, tool, text, expected):
        guess = tool.detect_language([("d", text)], self.LANGUAGES)
        assert (guess.language, guess.method) == (expected, "stopwords")
        assert guess.per_document == {"d": expected}

    def test_documents_weigh_by_length(self, tool):
        guess = tool.detect_language([("long", GERMAN_TEXT * 5), ("short", ENGLISH_TEXT)], self.LANGUAGES)
        assert guess.language == "de"
        assert guess.per_document == {"long": "de", "short": "en"}

    def test_too_few_stopwords_tell_nothing(self, tool):
        guess = tool.detect_language([("d", "Laterne Ballon Papier Laterne")], self.LANGUAGES)
        assert guess.language is None
        assert guess.per_document == {"d": None}

    def test_a_language_without_stopwords_is_undetectable(self, tool):
        guess = tool.detect_language([("d", ENGLISH_TEXT)], frozenset({"en", "fr"}))
        assert (guess.language, guess.method) == (None, "undetectable")


class TestDetectLanguageWithLingua:
    def test_the_lingua_scorer_is_preferred(self, tool, monkeypatch):
        monkeypatch.setattr(tool, "_lingua_scorer", lambda languages: (lambda text: {"de": 0.9, "en": 0.1}))
        guess = tool.detect_language([("d", ENGLISH_TEXT)], frozenset({"en", "de"}))
        assert (guess.language, guess.method) == ("de", "lingua")
        assert guess.scores == pytest.approx({"de": 0.9, "en": 0.1})

    def test_real_lingua(self, tool):
        pytest.importorskip("lingua")
        guess = tool.detect_language(
            [("d", "Die Katze sitzt auf dem Dach und schaut in den Garten hinunter.")],
            frozenset({"en", "de"}),
        )
        assert (guess.language, guess.method) == ("de", "lingua")


@pytest.fixture
def repo(tool, tmp_path, monkeypatch):
    """A repo root with an English and a German concept set."""
    root = tmp_path / "repo"
    write_lines(write_manifest(root / "concepts", "en") / "object.txt", "lantern")
    write_lines(write_manifest(root / "Konzepte", "de") / "object.txt", "Laterne")
    monkeypatch.setattr(tool, "PROJECT_ROOT", root)
    return root


def run_auto(tool, tmp_path, text, *extra):
    doc = write_lines(tmp_path / "doc.txt", text)
    out = tmp_path / "out.json"
    code = tool.main([str(doc), "--json", str(out), *extra])
    result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return code, result


@pytest.mark.usefixtures("no_blacklist", "no_lingua")
class TestMainLanguage:
    def test_a_german_document_uses_the_german_set(self, tool, tmp_path, repo):
        code, result = run_auto(tool, tmp_path, GERMAN_TEXT)
        assert code == 0
        assert result["stats"]["language"] == "de"
        assert Path(result["stats"]["concept_set"]) == repo / "Konzepte"
        found = by_concept(result)
        assert found["Laterne"]["tier"] == "exact"
        assert found["Ballon"]["tier"] == "new"

    def test_german_stopwords_bound_german_phrases(self, tool, tmp_path, repo):
        _code, result = run_auto(tool, tmp_path, GERMAN_TEXT)
        assert "der Ballon" not in by_concept(result)

    def test_an_english_document_uses_the_english_set(self, tool, tmp_path, repo):
        _code, result = run_auto(tool, tmp_path, ENGLISH_TEXT)
        assert result["stats"]["language"] == "en"
        assert Path(result["stats"]["concept_set"]) == repo / "concepts"
        assert by_concept(result)["lantern"]["tier"] == "exact"

    def test_language_overrides_detection(self, tool, tmp_path, repo):
        _code, result = run_auto(tool, tmp_path, GERMAN_TEXT, "--language", "en")
        assert (result["stats"]["language"], result["stats"]["language_method"]) == ("en", "--language")
        assert Path(result["stats"]["concept_set"]) == repo / "concepts"

    def test_concepts_dir_overrides_the_choice_of_set(self, tool, tmp_path, repo):
        _code, result = run_auto(tool, tmp_path, ENGLISH_TEXT, "--concepts-dir", str(repo / "Konzepte"))
        assert (result["stats"]["language"], result["stats"]["language_method"]) == ("de", "manifest")

    def test_an_undetectable_language_falls_back_to_english(self, tool, tmp_path, repo):
        _code, result = run_auto(tool, tmp_path, "Laterne Ballon. Laterne Ballon.")
        assert result["stats"]["language"] == "en"
        assert Path(result["stats"]["concept_set"]) == repo / "concepts"

    def test_a_language_without_a_set_is_an_error(self, tool, tmp_path, repo):
        code, result = run_auto(tool, tmp_path, ENGLISH_TEXT, "--language", "fr")
        assert code == 1 and result is None

    def test_an_invalid_language_code_is_refused(self, tool, tmp_path, repo):
        code, _result = run_auto(tool, tmp_path, ENGLISH_TEXT, "--language", "klingon")
        assert code == 2

    def test_no_sets_and_no_concepts_dir_is_an_error(self, tool, tmp_path, monkeypatch):
        empty = tmp_path / "empty_repo"
        empty.mkdir()
        monkeypatch.setattr(tool, "PROJECT_ROOT", empty)
        code, _result = run_auto(tool, tmp_path, ENGLISH_TEXT)
        assert code == 1

    def test_export_into_any_set_is_refused(self, tool, tmp_path, repo):
        code, _result = run_auto(tool, tmp_path, ENGLISH_TEXT, "--export-dir", str(repo / "Konzepte"))
        assert code == 2
        assert (repo / "Konzepte" / "object.txt").read_text(encoding="utf-8") == "Laterne\n"
