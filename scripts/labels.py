"""Read open_labels.xlsx (stdlib only) -> {"X-1": {"violation": True, "sec": 22.0}, ...}.

File names mix Latin "X" and Cyrillic "Х"; keys are normalized to Latin.
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def norm(name: str) -> str:
    return name.replace("Х", "X").replace("х", "x").strip().removesuffix(".mov")


def parse_time(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+)\s*мин\w*\s*(\d+)", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"(\d+)", text)
    return float(m.group(1)) if m else None


def read_labels(xlsx_path: str) -> dict:
    z = zipfile.ZipFile(xlsx_path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    rows = []
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    for row in sheet.iter(f"{{{NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", NS)
            inline = c.find("m:is", NS)
            if c.get("t") == "s" and v is not None:
                cells[col] = shared[int(v.text)]
            elif inline is not None:
                cells[col] = "".join(t.text or "" for t in inline.iter(f"{{{NS['m']}}}t"))
            elif v is not None:
                cells[col] = v.text
        rows.append(cells)
    out = {}
    for r in rows[1:]:
        if "A" not in r:
            continue
        out[norm(r["A"])] = {
            "violation": "без" not in r.get("C", "").lower(),
            "sec": parse_time(r.get("B", "")),
        }
    return out


if __name__ == "__main__":
    for k, v in read_labels(sys.argv[1]).items():
        print(k, v)
