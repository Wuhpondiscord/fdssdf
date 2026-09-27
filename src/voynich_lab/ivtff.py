"""IVTFF-aware parsing for Voynich Structure Lab.

The parser keeps manuscript text separate from IVTFF metadata and preserves
boundary uncertainty. It intentionally implements the subset needed by the
analysis pipeline rather than silently treating markup as manuscript glyphs.

IVTFF 2.0/2.0.2 page headers can carry page variables such as:
    <f1r> <! $Q=A $P=A $I=H $L=A $H=1 >
where $Q is quire, $L Currier language, $H hand, and $I illustration type.

Within locus text, the separator conventions used here are:
    .    certain word space
    ,    uncertain/potential word space
    <->  drawing-interrupted word space
    <~>  vertical-misalignment word space

Alternative readings in [a:b] resolve to the first reading for the primary
analysis stream, while uncertainty is retained on the containing token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import re
from pathlib import Path

COMMENT_LINE_RE = re.compile(r"^\s*#")
INLINE_COMMENT_RE = re.compile(r"<!.*?>")
PAGE_RE = re.compile(r"^<(?P<folio>[^.,;>]+)>\s*(?:<!\s*(?P<meta>.*?)\s*>)?\s*$")
LOCUS_RE = re.compile(
    r"^<(?P<folio>[^.,;>]+)\.(?P<line>[^,;>]+)(?:[,;](?P<extra>[^>]*))?>\s*(?P<text>.*)$"
)
META_RE = re.compile(r"\$(?P<key>[A-Za-z])=(?P<value>[^\s>]+)")
SEPARATOR_MARKERS = (("<->", "drawing"), ("<~>", "vertical"), (".", "certain"), (",", "uncertain"))


@dataclass
class Token:
    text: str
    had_alternates: bool = False
    had_illegible: bool = False


@dataclass
class PageMetadata:
    folio: str
    variables: dict[str, str] = field(default_factory=dict)

    @property
    def quire(self) -> str | None:
        return self.variables.get("Q")

    @property
    def currier_language(self) -> str | None:
        return self.variables.get("L")

    @property
    def hand(self) -> str | None:
        return self.variables.get("H")

    @property
    def illustration_type(self) -> str | None:
        return self.variables.get("I")


@dataclass
class LineRecord:
    folio: str
    line_no: str
    transcriber: str | None
    locus_code: str | None
    tokens: list[Token]
    separators: list[str]
    quire: str | None = None
    currier_language: str | None = None
    hand: str | None = None
    illustration_type: str | None = None


@dataclass
class ParsedDocument:
    lines: list[LineRecord] = field(default_factory=list)
    pages: dict[str, PageMetadata] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def conventional_tokens(self) -> list[str]:
        return [tok.text for line in self.lines for tok in line.tokens if tok.text]

    def assign_quires(self, quire_map: dict[str, str], *, overwrite: bool = False) -> None:
        """Fill quire values from an external folio->quire mapping.

        Native $Q metadata wins by default. Set overwrite=True only when the
        external map is known to be authoritative for the exact transcription
        version being analysed.
        """
        for line in self.lines:
            if line.folio in quire_map and (overwrite or not line.quire):
                line.quire = quire_map[line.folio]

    def quires_present(self) -> list[str]:
        return sorted({line.quire for line in self.lines if line.quire})

    def split_by_quire(self, held_out: str) -> tuple["ParsedDocument", "ParsedDocument"]:
        train = ParsedDocument(
            lines=[l for l in self.lines if l.quire != held_out], pages=self.pages.copy(), warnings=self.warnings.copy()
        )
        test = ParsedDocument(
            lines=[l for l in self.lines if l.quire == held_out], pages=self.pages.copy(), warnings=self.warnings.copy()
        )
        return train, test

    def to_analysis_text(self, uncertain_policy: str = "split") -> str:
        """Return manuscript text only, with IVTFF markup removed.

        uncertain_policy:
          - split: uncertain separators are represented as spaces.
          - merge: uncertain separators are erased.
        Drawing/vertical separators remain boundaries in both modes.
        """
        if uncertain_policy not in {"split", "merge"}:
            raise ValueError("uncertain_policy must be 'split' or 'merge'")
        rendered: list[str] = []
        for line in self.lines:
            if not line.tokens:
                continue
            parts = [line.tokens[0].text]
            for sep, tok in zip(line.separators, line.tokens[1:]):
                if sep == "uncertain" and uncertain_policy == "merge":
                    parts.append(tok.text)
                else:
                    parts.extend([" ", tok.text])
            rendered.append("".join(parts))
        return "\n".join(rendered)

    def text_by_quire(self, uncertain_policy: str = "split") -> dict[str, str]:
        grouped: dict[str, list[LineRecord]] = {}
        for line in self.lines:
            if not line.quire:
                continue
            grouped.setdefault(line.quire, []).append(line)
        out: dict[str, str] = {}
        for quire, lines in grouped.items():
            out[quire] = ParsedDocument(lines=lines).to_analysis_text(uncertain_policy=uncertain_policy)
        return out

    def glyph_stream_with_boundaries(self) -> tuple[str, list[str]]:
        """Return a continuous glyph stream and one gap label per glyph gap."""
        chars: list[str] = []
        gaps: list[str] = []
        for line_index, line in enumerate(self.lines):
            if not line.tokens:
                continue
            if chars and line_index > 0:
                gaps.append("line_break")
            for ti, tok in enumerate(line.tokens):
                for ci, ch in enumerate(tok.text):
                    if chars:
                        if ci > 0:
                            gaps.append("none")
                        elif ti > 0:
                            gaps.append(line.separators[ti - 1])
                    chars.append(ch)
        target = max(0, len(chars) - 1)
        if len(gaps) > target:
            gaps = gaps[:target]
        elif len(gaps) < target:
            gaps.extend(["none"] * (target - len(gaps)))
        return "".join(chars), gaps


def is_ivtff(raw_text: str) -> bool:
    for line in (raw_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#=IVTFF"):
            return True
        if PAGE_RE.match(s) or LOCUS_RE.match(s):
            return True
    return False


def load_quire_map(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().lower() == "folio":
                continue
            if len(row) < 2:
                continue
            folio, quire = row[0].strip(), row[1].strip()
            if folio:
                mapping[folio] = quire
    return mapping


def _resolve_alternate(raw: str) -> tuple[str, bool]:
    if raw.startswith("[") and raw.endswith("]"):
        options = raw[1:-1].split(":")
        return (options[0] if options else ""), True
    return raw, False


def _split_text_into_tokens(text: str) -> tuple[list[Token], list[str]]:
    tokens: list[Token] = []
    separators: list[str] = []
    buf: list[str] = []
    had_alt = False
    had_illegible = False

    def flush() -> None:
        nonlocal buf, had_alt, had_illegible
        if buf:
            tokens.append(Token("".join(buf), had_alternates=had_alt, had_illegible=had_illegible))
        buf = []
        had_alt = False
        had_illegible = False

    i = 0
    while i < len(text):
        matched_sep = None
        for marker, label in SEPARATOR_MARKERS:
            if text.startswith(marker, i):
                matched_sep = (marker, label)
                break
        if matched_sep is not None:
            marker, label = matched_sep
            flush()
            if tokens:
                if len(separators) < len(tokens):
                    separators.append(label)
                elif separators:
                    separators[-1] = label
            i += len(marker)
            continue
        if text[i] == "[":
            end = text.find("]", i + 1)
            if end != -1:
                resolved, alt = _resolve_alternate(text[i : end + 1])
                buf.append(resolved)
                had_alt = had_alt or alt
                had_illegible = had_illegible or ("?" in resolved)
                i = end + 1
                continue
        ch = text[i]
        if ch == "?":
            had_illegible = True
        buf.append(ch)
        i += 1
    flush()
    needed = max(0, len(tokens) - 1)
    if len(separators) > needed:
        separators = separators[:needed]
    elif len(separators) < needed:
        separators.extend(["certain"] * (needed - len(separators)))
    return tokens, separators


def _parse_locus_extra(extra: str) -> tuple[str | None, str | None]:
    extra = (extra or "").strip()
    if not extra:
        return None, None
    parts = extra.split(";")
    code = parts[0] or None
    transcriber = parts[-1] if len(parts) > 1 and parts[-1] else None
    return code, transcriber


def parse_ivtff(raw_text: str, *, strict: bool = False) -> ParsedDocument:
    doc = ParsedDocument()
    current_page: PageMetadata | None = None
    for line_number, raw_line in enumerate((raw_text or "").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or COMMENT_LINE_RE.match(raw_line):
            continue
        page_match = PAGE_RE.match(stripped)
        if page_match:
            folio = page_match.group("folio")
            meta = page_match.group("meta") or ""
            variables = {m.group("key").upper(): m.group("value") for m in META_RE.finditer(meta)}
            current_page = PageMetadata(folio=folio, variables=variables)
            doc.pages[folio] = current_page
            continue
        cleaned = INLINE_COMMENT_RE.sub("", stripped).strip()
        if not cleaned:
            continue
        m = LOCUS_RE.match(cleaned)
        if not m:
            msg = f"Unrecognized non-locus line {line_number}: {stripped[:80]}"
            if strict:
                raise ValueError(msg)
            doc.warnings.append(msg)
            continue
        folio = m.group("folio")
        if current_page is None or current_page.folio != folio:
            current_page = doc.pages.get(folio, PageMetadata(folio=folio))
            doc.pages.setdefault(folio, current_page)
        locus_code, transcriber = _parse_locus_extra(m.group("extra") or "")
        text = m.group("text").strip()
        if not text:
            continue
        tokens, separators = _split_text_into_tokens(text)
        tokens = [t for t in tokens if t.text]
        if not tokens:
            continue
        separators = separators[: max(0, len(tokens) - 1)]
        doc.lines.append(LineRecord(
            folio=folio,
            line_no=m.group("line"),
            transcriber=transcriber,
            locus_code=locus_code,
            tokens=tokens,
            separators=separators,
            quire=current_page.quire,
            currier_language=current_page.currier_language,
            hand=current_page.hand,
            illustration_type=current_page.illustration_type,
        ))
    return doc
