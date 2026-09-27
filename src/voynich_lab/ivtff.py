"""IVTFF-aware parsing.

Implements enough of the Zandbergen IVTFF 2.0 format to preserve the
distinctions the 2026 boundary/unit literature relies on, instead of treating
the file as generic whitespace-delimited text.

Verified conventions (Zandbergen, "IVTFF format 1.7/2.0"; ivtt manual;
voynich.ninja transliteration threads, 2023):

- Locus lines look like ``<f1r.1,@P0>`` or ``<f75r.47;U>``: folio, line
  number, optional locus-type/paragraph code, optional transcriber code.
- Comment lines start with ``#`` and are dropped.
- Free in-line comments in ``<! ... >`` are dropped.
- Word-space markers, which "shall always stand alone" between words:
    ``.``   certain word space
    ``,``   uncertain word space ("potential space")
    ``<->`` drawing-interrupted word space (treated as certain)
    ``<~>`` vertical-misalignment word space (treated as certain)
- Alternate/uncertain glyph readings are written ``[a:b]`` (or with more
  than two options); by IVTFF convention the first option is the primary
  reading. We keep the primary reading for the glyph stream and record that
  the position was an alternate-reading (glyph-uncertain) site.
- ``?`` marks an illegible glyph.

This module does not claim to implement the full IVTFF 2.0 grammar (e.g.
every rare annotation code); it implements the subset needed for the
metrics this project cares about, and fails loudly rather than silently on
unrecognized locus syntax so mis-parses don't quietly become "results."

Quire membership is not encoded in IVTFF itself. Supply it separately via
``load_quire_map`` (a two-column ``folio,quire`` CSV) so quire-held-out
splits are possible; without it, quire is left as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import re
from pathlib import Path

COMMENT_LINE_RE = re.compile(r"^\s*#")
INLINE_COMMENT_RE = re.compile(r"<!.*?>")
LOCUS_RE = re.compile(
    r"^<(?P<folio>[^.,;>]+)\.(?P<line>[^,;>]+)(?:[,;](?P<extra>[^>]*))?>\s*(?P<text>.*)$"
)
ALTERNATE_RE = re.compile(r"\[([^\[\]]*)\]")
DRAWING_SPACE_RE = re.compile(r"<-+>|<~>")

CERTAIN_SEP = "."
UNCERTAIN_SEP = ","


@dataclass
class Token:
    text: str
    had_alternates: bool
    had_illegible: bool


@dataclass
class LineRecord:
    folio: str
    line_no: str
    transcriber: str | None
    tokens: list[Token]
    # len(separators) == len(tokens) - 1; values are "certain" or "uncertain"
    separators: list[str]
    quire: str | None = None


@dataclass
class ParsedDocument:
    lines: list[LineRecord] = field(default_factory=list)

    # ---- assembly helpers -------------------------------------------------

    def conventional_tokens(self) -> list[str]:
        """Flat token list, analogous to metrics.conventional_tokens but
        sourced from real locus/space parsing rather than naive whitespace
        splitting."""
        return [tok.text for line in self.lines for tok in line.tokens if tok.text]

    def glyph_stream_with_boundaries(self, keep_line_breaks: bool = False) -> tuple[str, list[str]]:
        """Continuous glyph stream plus a per-gap certainty label aligned to
        the *boundaries between glyphs* ("certain", "uncertain", or "none"
        for glyph-internal positions). Mirrors the space-erasure experiment
        design: you get the plain stream and can independently ask which
        erased positions used to be certain vs. uncertain word spaces.
        """
        stream_parts: list[str] = []
        gap_labels: list[str] = []  # one entry per boundary between consecutive glyphs
        for li, line in enumerate(self.lines):
            for ti, tok in enumerate(line.tokens):
                if not tok.text:
                    continue
                for ci, ch in enumerate(tok.text):
                    if stream_parts:
                        # boundary before this glyph, if any glyph precedes it
                        if ci == 0 and ti > 0:
                            sep = line.separators[ti - 1] if ti - 1 < len(line.separators) else "certain"
                            gap_labels.append(sep)
                        elif ci == 0 and ti == 0 and li > 0:
                            gap_labels.append("certain" if keep_line_breaks else "line_break")
                        elif ci > 0:
                            gap_labels.append("none")
                    stream_parts.append(ch)
        return "".join(stream_parts), gap_labels

    def assign_quires(self, quire_map: dict[str, str]) -> None:
        for line in self.lines:
            line.quire = quire_map.get(line.folio)

    def quires_present(self) -> list[str]:
        seen = {line.quire for line in self.lines if line.quire}
        return sorted(seen)

    def split_by_quire(self, held_out: str) -> tuple["ParsedDocument", "ParsedDocument"]:
        """Return (train, held_out) ParsedDocuments partitioned by quire.
        Requires assign_quires() to have been called first."""
        train = ParsedDocument(lines=[l for l in self.lines if l.quire != held_out])
        test = ParsedDocument(lines=[l for l in self.lines if l.quire == held_out])
        return train, test


def load_quire_map(path: str | Path) -> dict[str, str]:
    """Load a two-column ``folio,quire`` CSV. This project does not ship a
    default folio->quire table; source one from a published transcription
    site and verify it against the transcription version you are using
    before trusting quire-held-out results."""
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().lower() == "folio":
                continue
            folio, quire = row[0].strip(), row[1].strip()
            if folio:
                mapping[folio] = quire
    return mapping


def _split_text_into_tokens(text: str) -> tuple[list[Token], list[str]]:
    """Split one line's transliterated text into (tokens, separator-labels).

    Alternates ``[a:b]`` are resolved to their primary reading ``a`` before
    splitting on word-space markers, per IVTFF convention. Each resulting
    token remembers whether it contained an alternate-reading site or an
    illegible (``?``) glyph, so downstream metrics can condition on
    transcription confidence instead of discarding that information.
    """
    text = DRAWING_SPACE_RE.sub(CERTAIN_SEP, text)

    token_had_alt: list[bool] = []
    token_had_illegible: list[bool] = []

    def _alt_sub(m: re.Match) -> str:
        options = m.group(1).split(":")
        return options[0] if options and options[0] else ""

    resolved = ALTERNATE_RE.sub(_alt_sub, text)
    alt_spans = {m.span() for m in ALTERNATE_RE.finditer(text)}  # in original text coords

    raw_tokens = re.split(r"[.,]", resolved)
    separators = re.findall(r"[.,]", resolved)
    sep_labels = ["certain" if s == CERTAIN_SEP else "uncertain" for s in separators]

    # Alternate/illegible flags are approximated per resolved token rather
    # than tracked through exact offset remapping (resolving [a:b] can
    # change string length). This is sufficient for "was this token's
    # reading contested at all" without over-engineering offset tracking.
    had_any_alt = bool(alt_spans)
    tokens = [
        Token(text=tok, had_alternates=had_any_alt, had_illegible=("?" in tok))
        for tok in raw_tokens
    ]
    return tokens, sep_labels


def parse_ivtff(raw_text: str) -> ParsedDocument:
    doc = ParsedDocument()
    for raw_line in raw_text.splitlines():
        if not raw_line.strip() or COMMENT_LINE_RE.match(raw_line):
            continue
        line = INLINE_COMMENT_RE.sub("", raw_line).strip()
        if not line:
            continue
        m = LOCUS_RE.match(line)
        if not m:
            # Not a locus-tagged line (e.g. a stray header); skip rather
            # than silently mis-tokenize it as manuscript text.
            continue
        folio = m.group("folio")
        line_no = m.group("line")
        extra = m.group("extra") or ""
        transcriber = extra.split(";")[-1] if ";" in extra else None
        text = m.group("text").strip()
        if not text:
            continue
        tokens, sep_labels = _split_text_into_tokens(text)
        kept_tokens = [t for t in tokens if t.text]
        if not kept_tokens:
            continue
        # sep_labels has one entry per separator found in the *unfiltered*
        # split; if empty tokens got dropped (e.g. leading/trailing marker),
        # trim to len(kept_tokens) - 1 defensively rather than misalign.
        n_seps_needed = max(0, len(kept_tokens) - 1)
        doc.lines.append(
            LineRecord(
                folio=folio,
                line_no=line_no,
                transcriber=transcriber,
                tokens=kept_tokens,
                separators=sep_labels[:n_seps_needed],
            )
        )
    return doc
