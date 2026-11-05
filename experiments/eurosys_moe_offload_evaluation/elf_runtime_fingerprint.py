#!/usr/bin/env python3
"""Fingerprint the loadable/runtime-relevant sections of an ELF64 file.

CUDA extension links are not byte-for-byte reproducible because NVCC can emit
random local symbol names into the non-allocated ``.strtab`` section.  Those
names and DWARF sections are never mapped into the process.  This helper keeps
the full-file SHA-256 as provenance while providing a deterministic gate over
every ``SHF_ALLOC`` section, including code, CUDA fatbins, dynamic symbols,
relocations, notes, and the size/layout of NOBITS sections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
ELF64_SECTION = struct.Struct("<IIQQQQIIQQ")
ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
SHF_ALLOC = 0x2
SHT_NOBITS = 8


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def c_string(table: bytes, offset: int) -> str:
    if not 0 <= offset < len(table):
        raise ValueError(f"section name offset is out of range: {offset}")
    end = table.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated section name")
    return table[offset:end].decode("utf-8", errors="strict")


def fingerprint(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    if len(blob) < ELF64_HEADER.size:
        raise ValueError("file is smaller than an ELF64 header")
    header = ELF64_HEADER.unpack_from(blob)
    ident = header[0]
    if ident[:4] != ELF_MAGIC:
        raise ValueError("file does not have an ELF magic value")
    if ident[4] != ELFCLASS64 or ident[5] != ELFDATA2LSB:
        raise ValueError("only little-endian ELF64 files are supported")

    section_offset = header[6]
    section_entry_size = header[11]
    section_count = header[12]
    section_names_index = header[13]
    if section_entry_size != ELF64_SECTION.size:
        raise ValueError(
            f"unexpected section header size: {section_entry_size}"
        )
    if section_count == 0 or not 0 <= section_names_index < section_count:
        raise ValueError("extended or invalid ELF section numbering is unsupported")
    table_end = section_offset + section_count * section_entry_size
    if section_offset <= 0 or table_end > len(blob):
        raise ValueError("section header table is out of file bounds")

    sections = [
        ELF64_SECTION.unpack_from(blob, section_offset + i * section_entry_size)
        for i in range(section_count)
    ]
    names_header = sections[section_names_index]
    names_offset, names_size = names_header[4], names_header[5]
    if names_offset + names_size > len(blob):
        raise ValueError("section name table is out of file bounds")
    names = blob[names_offset : names_offset + names_size]

    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        (
            name_offset,
            section_type,
            flags,
            address,
            file_offset,
            size,
            link,
            info,
            alignment,
            entry_size,
        ) = section
        if not flags & SHF_ALLOC:
            continue
        name = c_string(names, name_offset)
        if section_type == SHT_NOBITS:
            data = b""
        else:
            if file_offset + size > len(blob):
                raise ValueError(f"section {name!r} is out of file bounds")
            data = blob[file_offset : file_offset + size]
        metadata = {
            "index": index,
            "name": name,
            "type": section_type,
            "flags": flags,
            "address": address,
            "size": size,
            "link": link,
            "info": info,
            "alignment": alignment,
            "entry_size": entry_size,
        }
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        aggregate.update(struct.pack("<Q", len(encoded)))
        aggregate.update(encoded)
        aggregate.update(struct.pack("<Q", len(data)))
        aggregate.update(data)
        records.append({**metadata, "data_sha256": sha256_bytes(data)})

    if not records:
        raise ValueError("ELF has no SHF_ALLOC sections")
    return {
        "schema_version": 1,
        "algorithm": "sha256(canonical-metadata-and-bytes-of-all-SHF_ALLOC-sections)",
        "path": str(path.resolve()),
        "file_size": len(blob),
        "file_sha256": sha256_bytes(blob),
        "runtime_elf_sha256": aggregate.hexdigest(),
        "sections": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = fingerprint(args.elf)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["runtime_elf_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
