"""Ref2VA shot boundaries and generation identity in the v12 latent container."""
from contextlib import contextmanager, ExitStack
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading


MOTION = "incoming_motion_context"
SHOT_DECODE_VERSION = 1


@lru_cache(maxsize=None)
def _cache_lock(path):
    return threading.RLock()


@contextmanager
def cache_access(*paths):
    # The HTTP thread can switch cache owners while the executor is sampling
    # or decoding. Refuse a busy handoff before staging or replacing anything.
    with ExitStack() as stack:
        for path in sorted({str(Path(p).resolve()) for p in paths}):
            lock = _cache_lock(path)
            if not lock.acquire(blocking=False):
                raise RuntimeError("Ref2VA cache is in use. Retry after the current operation finishes.")
            stack.callback(lock.release)
        yield


def card_motion(card):
    # Workflow widgets from older versions have no switch. Only JSON false
    # opts out; strings, numbers, and null take the same legacy default in JS.
    return card.get(MOTION) is not False


def _integer(desc, key, default=None):
    value = desc.get(key, default)
    if type(value) is not int or value < 0:
        raise ValueError(f"Missing or invalid Ref2VA {key}: {value!r}")
    return value


def recorded_motion(desc, index):
    """Validate causal geometry before selecting standalone or paired decode."""
    frames = _integer(desc, "frames")
    trim = _integer(desc, "trim_frames")
    source = _integer(desc, "source_frames", frames)
    offset = _integer(desc, "visible_offset", 0)
    shape = desc.get("video", {}).get("shape")
    if shape is not None:
        t = shape[-3]
        if t < 2 or t % 5 != 2 or source != 5 + 17 * ((t - 2) // 5):
            raise ValueError("Ref2VA source frame count disagrees with video latent geometry")
    if not frames or offset + frames > source or trim >= frames:
        raise ValueError("Invalid Ref2VA visible window or overlap")
    if trim and (not index or trim < 5 or (trim - 5) % 17 or offset or frames != source):
        raise ValueError("Invalid Ref2VA continuation trim or visible window")
    flag = desc.get(MOTION, bool(trim) if index else True)
    if type(flag) is not bool:
        raise ValueError(f"Invalid recorded {MOTION}: {flag!r}")
    if index and flag != bool(trim):
        raise ValueError("Ref2VA incoming motion disagrees with trim_frames")
    if desc.get("continued_from_source") and (index or not flag or not offset):
        raise ValueError("Invalid Clip 0 continuation metadata")
    return flag


def clip_identity(manifest, index):
    desc = manifest["segments"][index]
    order = manifest.get("extender_clip_ids") or []
    return desc.get("clip_id") or (order[index] if index < len(order) else None)


def generation_identity(path, desc):
    if desc.get("generation_id"):
        return desc["generation_id"]
    # Only legacy generations need this read. Hash both streams, including
    # their tensor interpretation. Packing offsets can change on compaction.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for key in ("video", "audio"):
            spec = desc[key]
            digest.update(json.dumps([spec["dtype"], spec["shape"], spec["nbytes"]]).encode())
            stream.seek(spec["offset"])
            remaining = spec["nbytes"]
            while remaining > 0:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("Incomplete Ref2VA latent payload")
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()


def parent_identity(path, manifest, index):
    if index:
        return generation_identity(path, manifest["segments"][index - 1])
    source = manifest.get("source_video") or {}
    return source.get("id")


def reuse_problem(path, manifest, index, card):
    segments = (manifest or {}).get("segments", [])
    if index >= len(segments):
        return "disk cache is missing"
    desc = segments[index]
    if clip_identity(manifest, index) != card["id"]:
        return "cached clip ID does not match this card"
    incoming = recorded_motion(desc, index)
    if incoming != card_motion(card):
        return "cached incoming motion differs from the card setting"
    if not index and manifest.get("source_video") and incoming != bool(desc.get("continued_from_source")):
        return "cached Clip 0 conditioning differs"
    if incoming and "parent_generation_id" in desc:
        if desc["parent_generation_id"] != parent_identity(path, manifest, index):
            return "cached predecessor generation differs"
    return None


def restore_legacy_cards(cards, raw, manifest):
    # Saved pre-switch cards can inherit an actual cut from their matching
    # generation. Never overwrite an explicit user setting.
    state = json.loads(raw or "{}")
    original = state.get("clips", []) if isinstance(state, dict) else state
    if not isinstance(original, list):
        return
    for i, (card, old, desc) in enumerate(zip(cards, original, (manifest or {}).get("segments", []))):
        if isinstance(old, dict) and MOTION not in old and card["id"] == clip_identity(manifest, i):
            card[MOTION] = recorded_motion(desc, i)


def check_validated(path, manifest, cards):
    for i, card in enumerate(cards):
        if not card["validated"]:
            break
        problem = reuse_problem(path, manifest, i, card)
        if problem:
            raise ValueError(f"Validated Ref2VA clip {i + 1}: {problem}. Unvalidate it to generate a retake.")


def restore_causal_geometry(desc, index):
    """Undo the random-access representation, keeping a Clip 0 crop intact."""
    if "source_trim_frames" in desc:
        if _integer(desc, "trim_frames") != 0:
            raise ValueError("Independent Ref2VA descriptors must have zero trim")
        trim = _integer(desc, "source_trim_frames")
        if not index and desc.get("continued_from_source") and desc.get("visible_offset") == 0:
            # v3.1.1 overwrote Clip 0's crop with a zero trim on ON -> OFF.
            t = desc["video"]["shape"][-3]
            full = 5 + 17 * ((t - 2) // 5)
            if desc.get("source_frames") == desc["frames"] and full > desc["frames"]:
                desc["source_frames"] = full
                desc["visible_offset"] = full - desc["frames"]
        if trim:
            frames = _integer(desc, "source_frames")
            if (not index or _integer(desc, "visible_offset") != trim
                    or _integer(desc, "frames") != frames - trim):
                raise ValueError("Saved Ref2VA overlap and visible length disagree")
            desc["frames"] = frames
            desc["trim_frames"] = trim
            desc.pop("visible_offset")
            desc.pop("source_frames")
        desc.pop("source_trim_frames")
    desc[MOTION] = recorded_motion(desc, index)


def migrate_decoded_shots(manifest):
    """Invalidate only old seam-dependent derivatives. Keep raw offsets intact."""
    if manifest.get("sequence_mode", "ref2va") != "ref2va":
        return False
    changed = False
    repair_audio = False
    for i, desc in enumerate(manifest["segments"]):
        bridged_pair = bool(desc.get("source_trim_frames"))
        if "source_trim_frames" in desc:
            restore_causal_geometry(desc, i)
            changed = True
        incoming = recorded_motion(desc, i)
        if bridged_pair or (not incoming and (i or manifest.get("source_video"))):
            repair_audio = bridged_pair or desc.get("shot_decode_version") != SHOT_DECODE_VERSION
            if repair_audio:
                desc.pop("decoded_mp4_blob", None)
                desc.pop("decoded_seam_shift", None)
                desc.pop("final_video_cache_version", None)
                desc["shot_decode_version"] = SHOT_DECODE_VERSION
                changed = True
        if repair_audio:
            desc.pop("decoded_audio", None)
    if changed:
        manifest["shot_preview_needs_rebuild"] = True
        for key in ("preview_committed_count", "preview_audio_mode", "preview_portable_full"):
            manifest.pop(key, None)
    return changed


def branch_state(path, manifest):
    """Physical generation history, independent of file order and cache layout."""
    source = manifest.get("source_video") or {}
    root = [source.get(key) for key in ("id", "width", "height", "frame_count")]
    rows = []
    for i, desc in enumerate(manifest["segments"]):
        cid = clip_identity(manifest, i)
        if not cid:
            raise ValueError(f"Cannot identify cached clip {i + 1} for the motion handoff.")
        rows.append([cid, generation_identity(path, desc)])
    return [root, rows]


def check_handoff(source, target, common_base):
    if target is None:
        return
    if target == common_base:
        return
    root, rows = target
    if root == source[0] and rows == source[1][:len(rows)]:
        return
    raise ValueError("Ambiguous Ref2VA cache histories. Both caches were kept. "
                     "Save the active project before selecting a retake.")
