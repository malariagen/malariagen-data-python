"""Minimal, write-only BGZF (Blocked GZip Format) support.

BGZF is the compression format used throughout the SAM/BAM/VCF tooling
ecosystem (samtools, bcftools, tabix, bedtools, IGV, ...) for
compressed, indexable files. Structurally, it's just a sequence of
ordinary gzip members, each holding up to ~64KB of uncompressed data,
with a small extra header field recording each member's total size,
plus a fixed, empty final member used as an EOF marker.

Python's standard library `gzip.open()` produces valid gzip, but not
valid BGZF: tools that specifically expect BGZF for a `.vcf.gz` (or
similar) file can fail to read it correctly, or silently mis-parse it,
even though plain gzip decompresses just fine as a byte stream. This
module exists so that `.gz` output written by malariagen_data is
genuine BGZF, and therefore usable with the wider standard toolset
(bcftools, tabix, bedtools, etc.), not just tools that happen to
tolerate plain gzip.

See the BGZF section of the SAM specification for the on-disk format
this implements: https://samtools.github.io/hts-specs/SAMv1.pdf
"""

import struct
import zlib
from typing import Optional

# The fixed, empty BGZF block used to mark genuine end-of-file (as
# opposed to a truncated file). This exact byte sequence is part of the
# BGZF specification, not something we generate ourselves.
_BGZF_EOF_MARKER = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)

# Maximum amount of *uncompressed* data placed in a single BGZF block.
# Kept comfortably under the 16-bit BSIZE field's range (a block's
# total on-disk size, header and footer included, must fit in 65536
# bytes) even in the worst case where compression doesn't shrink the
# input at all; this matches the conservative block size used by other
# BGZF writers (e.g. Biopython's `Bio.bgzf`).
_MAX_BLOCK_UNCOMPRESSED_SIZE = 65280


def _compress_bgzf_block(data: bytes, level: int = 6) -> bytes:
    """Compress `data` (must be <= _MAX_BLOCK_UNCOMPRESSED_SIZE bytes)
    into a single, standalone BGZF block."""
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
    compressed = compressor.compress(data) + compressor.flush(zlib.Z_FINISH)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    isize = len(data) & 0xFFFFFFFF

    # Block layout: 18-byte header (with the BGZF "BC" extra field),
    # the compressed payload, then an 8-byte footer (CRC32 + ISIZE) —
    # the same footer any gzip member ends with.
    block_size = 18 + len(compressed) + 8
    header = struct.pack(
        "<BBBBIBBHBBHH",
        0x1F,
        0x8B,  # ID1, ID2: gzip magic number
        0x08,  # CM: compression method (deflate)
        0x04,  # FLG: FEXTRA set (an extra field follows)
        0,  # MTIME
        0,  # XFL
        0xFF,  # OS: unknown
        6,  # XLEN: length of the extra field below
        0x42,
        0x43,  # SI1, SI2: BGZF's "BC" extra subfield identifier
        2,  # SLEN: subfield length
        block_size - 1,  # BSIZE: total block size, minus 1
    )
    footer = struct.pack("<II", crc, isize)
    return header + compressed + footer


class BgzfWriter:
    """A minimal, write-only, text-mode BGZF writer.

    Used the same way as `gzip.open(path, "wt")`: construct, `write()`
    strings to it, and either call `close()` or use it as a context
    manager.
    """

    def __init__(self, path: str, encoding: str = "utf-8"):
        self._fh = open(path, "wb")
        self._encoding = encoding
        self._buffer = bytearray()
        self._closed = False

    def write(self, text: str) -> None:
        self._buffer.extend(text.encode(self._encoding))
        while len(self._buffer) >= _MAX_BLOCK_UNCOMPRESSED_SIZE:
            chunk = bytes(self._buffer[:_MAX_BLOCK_UNCOMPRESSED_SIZE])
            del self._buffer[:_MAX_BLOCK_UNCOMPRESSED_SIZE]
            self._fh.write(_compress_bgzf_block(chunk))

    def close(self) -> None:
        if self._closed:
            return
        if self._buffer:
            self._fh.write(_compress_bgzf_block(bytes(self._buffer)))
            self._buffer = bytearray()
        self._fh.write(_BGZF_EOF_MARKER)
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "BgzfWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def bgzf_open(path: str, mode: str = "wt", encoding: Optional[str] = "utf-8"):
    """Open `path` for BGZF-compressed text writing.

    Signature-compatible with `gzip.open(path, "wt")` for the subset of
    usage malariagen_data's VCF/text exporters need (write-only, text
    mode), so it can be used as a drop-in replacement for `gzip.open`
    wherever genuine BGZF (rather than plain gzip) output is required.
    """
    if mode not in ("wt", "w"):
        raise ValueError(
            f"Unsupported mode {mode!r}: bgzf_open() only supports writing text "
            "('wt' or 'w')."
        )
    return BgzfWriter(path, encoding=encoding or "utf-8")
