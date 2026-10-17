"""Video -> scene-state timeline (local VLM) -> rule engine -> violations with evidence.

Stage 1  decode frames with real timestamps (2 fps kept for evidence)
Stage 2  VLM reads sampled frames into a small state: people count, cashier present, phone, personal items
Stage 3  at every moment the cashier leaves, inspect what was left behind:
         objects on the desk (cash, bags, tablets...) and the laptop screen (zoomed crop: unlocked program vs lock screen)
Stage 4  rules over the timeline -> events; evidence frames + factual description + regulation from rules.yaml
"""
import base64
import logging
import re
import time
from pathlib import Path

import yaml

from .. import config
from ..video import Frame, probe, sample_frames, to_jpeg_bytes
from . import vlm

log = logging.getLogger("analyze")
REGISTRY = yaml.safe_load((Path(__file__).parent / "rules.yaml").read_text(encoding="utf-8"))

STATE_KEYS = ["people", "cashier_present", "phone", "personal_items", "eating"]

STATE_PROMPT = """Security camera frame of a small bank cash room: a desk with a laptop and a banknote counting machine, and a grey metal safe. The cashier is the bank employee working at that desk; other people are clients, collectors or visitors. Ignore people visible through the glass wall or in reflections.
Answer ONLY with JSON:
{"people": <number of people inside the room>, "cashier_present": <true/false>, "phone": <true if the cashier holds or looks at a mobile phone>, "personal_items": <true if a handbag, coat or jacket is on the desk or held by the cashier>, "eating": <true if the cashier is drinking from a cup or eating>}"""

LEFT_PROMPT = """Security camera frame of a bank cashier's workplace: a desk with a laptop, a desk phone and a banknote counting machine, and a grey metal safe.
List what lies on the desk. Answer ONLY with JSON:
{"objects_on_desk": [objects lying on the desk, e.g. banknotes, money bundle, bag, cup, tablet, mobile phone, documents, keys]}"""

LOCATE_LAPTOP = 'Locate the laptop in the image. Output only JSON: {"bbox_2d": [x1, y1, x2, y2]}'
SCREEN_PROMPT = ("This is a close-up of a laptop. Is its screen showing a bright program window (white areas, text, tables) "
                 "or is it dark / a lock-screen picture / off? Answer with one word: bright or dark.")

# Objects that normally stay on a cashier's desk.
STANDARD_EQUIPMENT = ("counting machine", "counter", "desk phone", "telephone", "laptop", "computer", "monitor",
                      "keyboard", "mouse", "printer", "scanner", "cable", "wire", "lamp", "machine",
                      "card reader", "terminal", "phone base", "speaker", "router", "charger", "nothing", "none", "safe", "cabinet", "drawer", "chair",
                      "outlet", "socket", "power", "adapter", "cord", "plug", "headset", "mouse pad", "stand", "screen",
                      "display", "sticker", "label", "logo", "desk surface",
                      "register", "device", "receipt")


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return False


def _small(img, side=640):
    im = img.copy()
    im.thumbnail((side, side))
    return im


def read_state(frame: Frame) -> dict:
    raw = vlm.ask([_small(frame.image)], STATE_PROMPT, max_tokens=60)
    js = vlm.parse_json(raw) or {}
    try:
        people = int(js.get("people"))
    except (TypeError, ValueError):
        people = None
    return {"t": frame.t, "people": people, "cashier_present": _as_bool(js.get("cashier_present")),
            "phone": _as_bool(js.get("phone")), "personal_items": _as_bool(js.get("personal_items")),
            "eating": _as_bool(js.get("eating")), "raw": raw}


LOCATE_DESK = 'Locate the desk (table top) on which the laptop stands. Output only JSON: {"bbox_2d": [x1, y1, x2, y2]}'
DESK_PROMPT = """This is a zoomed view of a bank cashier's desk. List only the objects you can clearly see lying on the desk surface. Do not guess; if the desk is empty apart from the equipment, say so.
Answer ONLY with JSON: {"objects_on_desk": ["..."]}"""


RU_OBJECTS = [("money bundle", "пачка денег"), ("banknotes", "купюры"), ("banknote", "купюра"), ("money", "деньги"),
              ("cash", "наличные"), ("bundle", "пачка"), ("mobile phone", "мобильный телефон"), ("smartphone", "смартфон"),
              ("tablet", "планшет"), ("cup", "чашка"), ("mug", "кружка"), ("bottle", "бутылка"), ("handbag", "сумка"),
              ("bag", "сумка"), ("envelope", "конверт"), ("documents", "документы"), ("document", "документ"),
              ("papers", "бумаги"), ("paper", "бумага"), ("keys", "ключи"), ("key", "ключ"), ("wallet", "кошелёк"),
              ("notebook", "блокнот"), ("folder", "папка"), ("box", "коробка"), ("coat", "верхняя одежда"),
              ("jacket", "куртка"), ("food", "еда"), ("drink", "напиток"), ("green object", "зелёный предмет"),
              ("white object", "белый предмет"), ("black object", "чёрный предмет"), ("object", "предмет")]


def to_russian(name: str) -> str:
    for en, ru in RU_OBJECTS:
        if en in name:
            return ru
    return name


def _crop(frame: Frame, box: list[float] | None, margin: float = 0.06, scale: int = 2):
    if not box:
        return frame.image
    W, H = frame.image.size
    x1, y1, x2, y2 = box
    c = frame.image.crop((int(max(0, x1 - margin) * W), int(max(0, y1 - margin) * H),
                          int(min(1, x2 + margin) * W), int(min(1, y2 + margin) * H)))
    return c.resize((c.width * scale, c.height * scale))


def check_left_behind(frame: Frame, extra: Frame | None = None) -> dict:
    """What stays at the workplace after the cashier left: items on the (zoomed) desk + laptop screen state.
    The later frame is read by the stronger judge model when it is available."""
    desk = vlm.parse_bbox(vlm.ask([frame.image], LOCATE_DESK, max_tokens=40))
    objs = []
    reads = [(frame, "scan")] + ([(extra, "judge" if vlm.available("judge") else "scan")] if extra is not None else [])
    for fr, kind in reads:
        js = vlm.parse_json(vlm.ask([_crop(fr, desk)], DESK_PROMPT, max_tokens=90, kind=kind))
        found = js.get("objects_on_desk") if isinstance(js, dict) else js
        objs += found if isinstance(found, list) else ([str(found)] if found else [])
    # Anything that is not standard desk equipment counts as left behind ("phone" alone = the desk phone).
    suspicious = []
    for o in objs:
        s = re.sub(r"\(.*?\)", "", str(o).lower()).strip()
        if s in ("phone", "telephone", "phones"):
            continue
        # Ordinary stationery is normal on a desk and is often hallucinated on an empty desk: ignore it.
        if re.search(r"\b(documents?|papers?|keys?|pens?|pencils?|scissors|stapler|notebooks?|folders?|sticky notes?)\b", s):
            continue
        ru = to_russian(s)
        if s and not any(w in s for w in STANDARD_EQUIPMENT) and ru not in suspicious:
            suspicious.append(ru)
    screen = None
    lap = vlm.parse_bbox(vlm.ask([frame.image], LOCATE_LAPTOP, max_tokens=40))
    if lap:
        screen = vlm.ask([_crop(frame, lap, margin=0.02, scale=4)], SCREEN_PROMPT, max_tokens=4).strip().lower()
    return {"objects": objs, "suspicious": suspicious, "screen_bright": bool(screen and "bright" in screen), "desk": desk}


# ---------------- rules over the timeline ----------------

def _smooth(vals: list[bool]) -> list[bool]:
    """Majority vote over a 3-sample window to remove single-frame VLM glitches."""
    out = []
    for i in range(len(vals)):
        w = vals[max(0, i - 1): i + 2]
        out.append(sum(w) * 2 > len(w))
    return out


def _runs(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    return runs


def find_departures(states: list[dict]) -> list[int]:
    """Indices i where the cashier was present at i-1 and is gone from i on (for at least 2 samples or to the end)."""
    present = _smooth([s["cashier_present"] and (s["people"] or 0) > 0 for s in states])
    deps = []
    for i in range(1, len(states)):
        if present[i - 1] and not present[i] and not any(present[i: i + 2]):
            deps.append(i)
    return deps


def detect_events(states: list[dict], frames: list[Frame], deadline: float) -> list[dict]:
    events = []
    if not states:
        return events
    ts = [s["t"] for s in states]

    # A. Cashier leaves: check what is left behind on an empty-room frame ~1.5 s after leaving.
    for i in find_departures(states):
        t_abs = ts[i]
        if time.time() > deadline:
            break
        f_after = min(frames, key=lambda f: abs(f.t - (t_abs + 1.5)))
        f_later = min(frames, key=lambda f: abs(f.t - (t_abs + 3.5)))
        lb = check_left_behind(f_after, f_later if f_later.t != f_after.t else None)
        tc = max(0.0, t_abs - 3.5)
        others = [s["people"] or 0 for s in states[max(0, i - 2): i]]
        log.info("departure at %.1fs: %s", t_abs, lb)
        if lb["suspicious"]:
            events.append({"id": "valuables_left", "t": tc, "conf": 0.75,
                           "detail": "кассир покидает рабочее место; на столе остаются: " + ", ".join(lb["suspicious"][:4]),
                           "evidence_t": [tc, f_after.t]})
        if lb["screen_bright"]:
            events.append({"id": "unlocked_workstation", "t": tc, "conf": 0.7,
                           "detail": "кассир покидает рабочее место; экран компьютера остаётся с открытой рабочей программой",
                           "evidence_t": [tc, f_after.t]})
        if others and min(others) >= 2:
            events.append({"id": "left_during_service", "t": tc, "conf": 0.55,
                           "detail": "кассир уходит с рабочего места, когда в помещении находится посетитель",
                           "evidence_t": [tc, f_after.t]})

    # B. Phone in the cashier's hands (2+ consecutive samples). Only when the cashier is alone in the room:
    #    with a client present the model cannot reliably tell whose phone it is (clients may use phones).
    for a, b in _runs(_smooth([s["phone"] and (s["people"] or 0) <= 1 for s in states]), 2):
        events.append({"id": "phone_use", "t": ts[a], "conf": 0.6,
                       "detail": "кассир держит в руках мобильный телефон",
                       "evidence_t": [ts[a], ts[min(b, a + 2)]]})

    # C. Personal items / eating — only while the cashier is alone (with visitors the model confuses
    #    their bags and cups with the cashier's), 3+ consecutive samples for items, 2+ for eating.
    alone = [(s["people"] or 0) <= 1 and s["cashier_present"] for s in states]
    for a, b in _runs(_smooth([s["personal_items"] and al for s, al in zip(states, alone)]), 3):
        events.append({"id": "personal_items", "t": ts[a], "conf": 0.55,
                       "detail": "на рабочем месте кассира находятся личные вещи (сумка или верхняя одежда)",
                       "evidence_t": [ts[a], ts[min(b, a + 2)]]})
    for a, b in _runs(_smooth([s["eating"] and al for s, al in zip(states, alone)]), 2):
        events.append({"id": "personal_items", "t": ts[a], "conf": 0.6,
                       "detail": "кассир ест или пьёт на рабочем месте во время работы с ценностями",
                       "evidence_t": [ts[a], ts[min(b, a + 2)]]})

    # D. Three or more people in the room (cashier + client + someone else).
    for a, b in _runs([(s["people"] or 0) >= 3 for s in states], 2):
        events.append({"id": "outsider_in_cash_area", "t": ts[a], "conf": 0.6,
                       "detail": f"в помещении одновременно находятся {max(s['people'] or 0 for s in states[a:b + 1])} человека",
                       "evidence_t": [ts[a], ts[min(b, a + 2)]]})

    # E. A visitor comes in and leaves within a few seconds (e.g. a collector hands over a bag and leaves
    #    before it is checked in his presence). Needs a stable "cashier alone" period before and after.
    cnt = [s["people"] or 0 for s in states]
    visit = [c >= 2 for c in cnt]
    for a, b in _runs(visit, 2):
        if a < 2 or b + 2 >= len(states):
            continue
        if not (all(c == 1 for c in cnt[a - 2:a]) and all(c == 1 for c in cnt[b + 1:b + 3])):
            continue
        dur = ts[b + 1] - ts[a]
        if dur <= 9.0:
            tc = ts[a] + (ts[b] - ts[a]) / 2
            events.append({"id": "collection_bag", "t": tc, "conf": 0.5,
                           "detail": f"посетитель входит, передаёт ценности и покидает помещение примерно через {dur:.0f} с; "
                                     "проверка целостности и пломбы в его присутствии на видео не наблюдается",
                           "evidence_t": [tc, ts[b]]})

    # F. Someone enters while banknotes lie openly on the desk (stable count change 1 -> 2+), at most 3 checks.
    checks = 0
    for i in range(2, len(states) - 1):
        if time.time() > deadline or checks >= 3:
            break
        if cnt[i - 2] == cnt[i - 1] == 1 and cnt[i] >= 2 and cnt[i + 1] >= 2 and states[i - 1]["cashier_present"]:
            checks += 1
            reads = []
            for dt in (0.0, 1.5):  # persistence: banknotes must be seen on both reads
                f_entry = min(frames, key=lambda f: abs(f.t - (ts[i] + dt)))
                js = vlm.parse_json(vlm.ask([f_entry.image], LEFT_PROMPT, max_tokens=80))
                found = js.get("objects_on_desk") if isinstance(js, dict) else None
                objs = [str(o).lower() for o in found] if isinstance(found, list) else []
                reads.append([to_russian(o) for o in objs
                              if any(w in o for w in ("banknote", "money", "cash", "bundle", "bill"))
                              and not any(w in o for w in ("machine", "counter", "bag", "sack", "register", "drawer"))])
            money = reads[0] if all(reads) else []
            log.info("entry at %.1fs: %s", ts[i], reads)
            if money:
                events.append({"id": "cash_exposed", "t": ts[i], "conf": 0.5,
                               "detail": "в помещение входит посетитель, при этом на столе открыто лежат: " + ", ".join(money[:3]),
                               "evidence_t": [ts[i], ts[min(i + 1, len(ts) - 1)]]})
    return events


def merge_events(events: list[dict], gap: float = 8.0) -> list[dict]:
    """Keep one event per type within `gap` seconds (highest confidence)."""
    events = sorted(events, key=lambda e: (e["id"], e["t"]))
    out = []
    for e in events:
        if out and out[-1]["id"] == e["id"] and e["t"] - out[-1]["t"] <= gap:
            if e["conf"] > out[-1]["conf"]:
                out[-1] = e
            continue
        out.append(e)
    return sorted(out, key=lambda e: e["t"])


# ---------------- judge (stronger model re-checks suspicious moments) ----------------

HYPOTHESIS = {
    "valuables_left": "The cashier walks away from the workplace and money or other items stay on the desk unattended.",
    "unlocked_workstation": "The cashier walks away and the laptop screen stays on with an open work program (not a lock screen).",
    "left_during_service": "The cashier walks away from the desk while a client is still in the room.",
    "phone_use": "The cashier (bank employee at the desk) holds or uses a mobile phone.",
    "personal_items": "The cashier has personal items (bag, coat) at the workplace, or eats or drinks while working with cash.",
    "outsider_in_cash_area": "An extra person, who is neither the cashier nor the client being served, is in the cash room near the cashier's desk.",
    "collection_bag": "A person brings a bag of valuables and leaves within seconds, before the cashier checks the bag and its seal in his presence.",
    "cash_exposed": "When a visitor comes in, banknotes are lying openly on the desk.",
    "cash_handling": "Cash is handed over without being counted in front of the client.",
    "other": "In these frames the cashier clearly breaks a cash-handling rule (for example leaves cash unattended, hands over cash without counting it, or lets an outsider handle cash).",
}
# For these types the scan already used a zoomed, targeted check; the judge refines but does not veto.
NO_VETO = {"valuables_left", "unlocked_workstation"}

JUDGE_PROMPT = """You are checking a possible cash-handling violation in a bank cash room. The images are consecutive security-camera frames (about 2 seconds apart, in time order).
Hypothesis: {hyp}
Look carefully and answer ONLY with JSON:
{{"confirmed": true or false, "plausibility": <0-100, how likely the hypothesis is true from what is visible>, "seen_ru": "<one short factual sentence in Russian: who does what and which objects are visible; no judgements or guesses>", "key_object": "<short English name of the person or object to mark, e.g. 'banknotes on the desk', 'laptop screen', 'mobile phone', 'cashier'>"}}"""

SECOND_LOOK_PROMPT = """These are {n} consecutive frames (about {gap:.0f} s apart) from a security camera in a bank cash room (a desk with a laptop and a banknote counting machine, a grey safe; the cashier is the bank employee at that desk).
Does the cashier break cash-handling rules in these frames? Possible problems:
a) walks away leaving money/items on the desk or the laptop unlocked
b) uses a mobile phone
c) eats or drinks, or keeps a bag or coat at the desk
d) an extra person (not the served client) approaches the cashier's desk
e) hands cash over without counting it
f) takes a bag of valuables from someone who leaves immediately
Answer ONLY with JSON: {{"v": true or false, "type": "<a-f or none>", "p": <0-100 plausibility>, "frame": <1-{n}>}}"""
SECOND_LOOK_TYPES = {"a": "other", "b": "phone_use", "c": "personal_items", "d": "outsider_in_cash_area",
                     "e": "cash_handling", "f": "collection_bag"}


def _ctx_frames(frames: list[Frame], t: float, offsets=(-2.0, 0.0, 2.5), side=560) -> list[Frame]:
    out, seen = [], set()
    for o in offsets:
        f = min(frames, key=lambda f: abs(f.t - (t + o)))
        if f.t not in seen:
            seen.add(f.t)
            out.append(f)
    return out


_OPINION = ("наруш", "норм", "правил", "не является", "намерен", "подозр", "безопасност", "запрещ", "должен", "должна",
            "следует", "вероятно", "возможно", "скорее всего")


def clean_seen(text: str) -> str:
    """Keep only factual sentences: the report must not contain unconfirmed conclusions."""
    import re as _re
    parts = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", text or "") if x.strip()]
    facts = [x for x in parts if not any(w in x.lower() for w in _OPINION)]
    seen, out = set(), []
    for x in facts:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return " ".join(out[:2])


def judge_event(ev: dict, frames: list[Frame]) -> dict:
    """Stronger model re-checks one event: confirm/reject, plausibility, description, box around the key object."""
    ctx = _ctx_frames(frames, ev["t"])
    raw = vlm.ask([_small(f.image, 560) for f in ctx], JUDGE_PROMPT.format(hyp=HYPOTHESIS.get(ev["id"], "")),
                  max_tokens=110, kind="judge")
    js = vlm.parse_json(raw)
    js = js if isinstance(js, dict) else {}
    try:
        plaus = max(0.0, min(100.0, float(js.get("plausibility", 50)))) / 100
    except (TypeError, ValueError):
        plaus = 0.5
    verdict = {"confirmed": _as_bool(js.get("confirmed")), "plausibility": plaus,
               "seen": clean_seen(str(js.get("seen_ru") or "")), "key_object": str(js.get("key_object") or "").strip()}
    # Box around the key object on the frame at the event moment.
    key = verdict["key_object"] or "the cashier"
    f0 = min(frames, key=lambda f: abs(f.t - ev["t"]))
    if ev["id"] in ("valuables_left", "unlocked_workstation"):
        f0 = min(frames, key=lambda f: abs(f.t - ev["evidence_t"][-1]))  # empty desk after leaving
    g = vlm.ask([f0.image], f'Locate {key} in the image. Output only JSON: {{"bbox_2d": [x1, y1, x2, y2]}}',
                max_tokens=40, kind="judge")
    verdict["box"] = {"t": round(f0.t, 2), "bbox": vlm.parse_bbox(g), "label": key}
    log.info("judge %s@%.1f: %s", ev["id"], ev["t"], {k: v for k, v in verdict.items() if k != "box"})
    return verdict


def second_look(frames: list[Frame], duration: float, deadline: float) -> list[dict]:
    """When the scan found nothing, the judge reviews the video in short windows (3 frames each)."""
    win = 6.0
    k_windows = max(1, min(4, int(round(duration / win))))
    span = duration / k_windows
    found = []
    for w in range(k_windows):
        if time.time() + 8 > deadline:
            break
        ts = [w * span + span * q for q in (0.15, 0.5, 0.85)]
        picks = [min(frames, key=lambda f: abs(f.t - t)) for t in ts]
        raw = vlm.ask([_small(f.image, 512) for f in picks], SECOND_LOOK_PROMPT.format(n=3, gap=span * 0.35),
                      max_tokens=40, kind="judge")
        js = vlm.parse_json(raw)
        js = js if isinstance(js, dict) else {}
        log.info("window %.1f-%.1fs: %s", w * span, (w + 1) * span, js)
        typ = SECOND_LOOK_TYPES.get(str(js.get("type", "")).strip().lower()[:1])
        try:
            plaus = float(js.get("p", 0)) / 100
            k = min(2, max(0, int(js.get("frame", 2)) - 1))
        except (TypeError, ValueError):
            continue
        if _as_bool(js.get("v")) and typ and plaus >= 0.8:
            t = picks[k].t
            found.append({"id": typ, "t": t, "conf": 0.6 * plaus, "detail": "на кадрах зафиксировано действие, требующее проверки",
                          "evidence_t": [t, picks[min(2, k + 1)].t], "from_second_look": True,
                          "letter": str(js.get("type", "")).strip().lower()[:1]})
    return found


# ---------------- sections of the video ----------------

def build_segments(states: list[dict], duration: float) -> list[dict]:
    present = _smooth([s["cashier_present"] and (s["people"] or 0) > 0 for s in states])
    kinds = []
    for s, pr in zip(states, present):
        n = s["people"] or 0
        kinds.append("absent" if not pr else "crowd" if n >= 3 else "service" if n >= 2 else "alone")
    labels = {"alone": "Кассир на рабочем месте один",
              "service": "Кассир и посетитель в помещении",
              "crowd": "В помещении 3 и более человек",
              "absent": "Кассира нет на рабочем месте"}
    segs = []
    for i, (s, k) in enumerate(zip(states, kinds)):
        start = 0.0 if i == 0 else s["t"]
        if segs and segs[-1]["kind"] == k:
            continue
        if segs:
            segs[-1]["end"] = start
        segs.append({"start": round(start, 2), "end": round(duration, 2), "kind": k, "label": labels[k]})
    for sg in segs:
        sg["end"] = round(max(sg["end"], sg["start"]), 2)
    return segs


# ---------------- output ----------------

def _nearest(frames: list[Frame], t: float) -> Frame:
    return min(frames, key=lambda f: abs(f.t - t))


def _annotated_jpeg(frame: Frame, box: list[float] | None, label: str) -> bytes:
    from PIL import ImageDraw
    img = frame.image.convert("RGB").copy()
    if box:
        W, H = img.size
        d = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        for k in range(3):
            d.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=(255, 59, 48))
    return to_jpeg_bytes(img, 848)


def build_violation(ev: dict, frames: list[Frame]) -> dict:
    spec = REGISTRY["violations"][ev["id"]]
    doc = REGISTRY["documents"].get(spec.get("doc") or "")
    j = ev.get("judge") or {}
    box = (j.get("box") or {})
    ev_frames, used = [], set()
    times = list(ev["evidence_t"][:3])
    if box.get("t") is not None and all(abs(box["t"] - t) > 0.4 for t in times):
        times = times[:2] + [box["t"]]
    for t in times:
        f = _nearest(frames, t)
        if f.t in used:
            continue
        used.add(f.t)
        has_box = box.get("bbox") and abs(f.t - box.get("t", -99)) < 0.3
        img = _annotated_jpeg(f, box["bbox"], box.get("label", "")) if has_box else to_jpeg_bytes(f.image, 848)
        ev_frames.append({"timestamp_sec": round(f.t, 2), "image_base64": base64.b64encode(img).decode()})
    tc = ev_frames[0]["timestamp_sec"]
    conf = ev["conf"]
    seen = ev["detail"]
    if j:
        conf = 0.4 * ev["conf"] + 0.6 * j["plausibility"] if j["confirmed"] else min(ev["conf"], 0.45)
        if j["seen"]:
            seen = f"{ev['detail']}. {j['seen']}" if ev["id"] in NO_VETO else j["seen"]
    conf = round(max(0.05, min(0.99, conf)), 2)
    basis = f"{doc['short']}, {spec['clause']}" if doc else "источник не определён, требуется проверка человеком"
    desc = (f"{tc:.1f} с: {seen.rstrip('.')}. {spec['explain']} Уверенность модели: {round(conf * 100)}%. "
            f"Основание: {basis}.")
    return {
        "type": spec["name"],
        "timecode_sec": tc,
        "description": desc,
        "evidence_frames": ev_frames,
        # extra fields for the UI only (the jury API returns exactly the required keys)
        "confidence": conf,
        "seen": seen,
        "why": spec["explain"],
        "boxes": [{"t": box["t"], "bbox": box["bbox"], "label": box.get("label", "")}] if box.get("bbox") else [],
        "escalated": bool(j),
        "judge_confirmed": bool(j and j.get("confirmed")),
        "regulation": ({"document": doc["short"], "clause": spec["clause"], "url": doc["url"]} if doc
                       else {"document": "Источник не определён", "clause": "требуется проверка человеком", "url": ""}),
    }


SEC_PER_FRAME = 3.4    # measured on MacBook Air M4 (Qwen3-VL-4B 4-bit, 640 px)
CHECKS_RESERVE = 18.0  # time kept for the "left behind" / entry checks
JUDGE_SEC = 14.0       # one judge call pair (verify + locate) with Qwen3-VL-8B


def analyze(path: Path, progress, budget: float | None = None) -> dict:
    t0 = time.time()
    budget = budget or config.TIME_BUDGET_SEC
    deadline = t0 + budget
    info = probe(path)
    progress(0.02, "чтение видео")
    frames = sample_frames(path, fps=2.0)
    if not frames:
        raise ValueError("no frames decoded")
    duration = max(info["duration"], frames[-1].t)
    use_judge = vlm.available("judge")

    # ~1 sample per 1.5 s; long videos / busy queue -> sparser sampling to stay within the time budget.
    reserve = CHECKS_RESERVE + (3 * JUDGE_SEC if use_judge else 0)
    max_frames = int(max(6, min(18, (budget - reserve) / SEC_PER_FRAME)))
    step = max(1.5, duration / max_frames)
    picked, next_t = [], 0.0
    for f in frames:
        if f.t + 1e-6 >= next_t:
            picked.append(f)
            next_t = f.t + step

    vlm.load("scan")
    states = []
    for i, f in enumerate(picked):
        if time.time() > deadline - reserve:
            log.warning("time budget hit after %d/%d frames", i, len(picked))
            break
        states.append(read_state(f))
        progress(0.05 + 0.7 * (i + 1) / len(picked), f"сканирование кадров {i + 1}/{len(picked)}")

    progress(0.78, "проверка рабочего места")
    events = merge_events(detect_events(states, frames, deadline - (2 * JUDGE_SEC if use_judge else 0)))

    stats = {"frames_scanned": len(states), "candidates": len(events), "escalations": 0, "confirmed": 0, "rejected": 0}
    if use_judge:
        progress(0.85, "проверка подозрительных моментов (8B)")
        if not events and time.time() + JUDGE_SEC < deadline:
            had_departure = bool(find_departures(states))
            events = []
            for ev in second_look(frames, duration, deadline - JUDGE_SEC):
                # The dedicated "cashier left" check already inspected every departure and found nothing.
                if ev["letter"] == "a" and had_departure:
                    log.info("second look 'walked away' at %.1f ignored: departure check found nothing", ev["t"])
                    continue
                events.append(ev)
            stats["candidates"] += len(events)
        kept = []
        for ev in sorted(events, key=lambda e: -e["conf"]):
            if time.time() + JUDGE_SEC > deadline:
                if not ev.get("from_second_look"):
                    kept.append(ev)  # no time left to judge: keep the scan result; unconfirmed second-look hints are dropped
                continue
            ev["judge"] = judge_event(ev, frames)
            stats["escalations"] += 1
            # Second-look findings need an explicit confirmation by the judge (a second, independent look).
            weak = (not ev["judge"]["confirmed"] or ev["judge"]["plausibility"] < 0.6) if ev.get("from_second_look") \
                else (not ev["judge"]["confirmed"] and ev["judge"]["plausibility"] < 0.35)
            if ev["id"] not in NO_VETO and weak:
                log.info("judge rejected %s@%.1f", ev["id"], ev["t"])
                stats["rejected"] += 1
                continue
            stats["confirmed"] += int(ev["judge"]["confirmed"])
            kept.append(ev)
        events = sorted(kept, key=lambda e: e["t"])

    violations = [build_violation(e, frames) for e in events]
    progress(0.98, "формирование отчёта")
    summary = (f"Проанализировано {len(states)} кадров из {duration:.0f} с видео"
               f"{'; подозрительные моменты перепроверены моделью Qwen3-VL-8B' if use_judge else ''}.")
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "summary": summary,
        "duration": round(duration, 2),
        "segments": build_segments(states, duration),
        "pipeline": stats,
        "states": [{k: s.get(k) for k in ["t"] + STATE_KEYS} for s in states],
    }
