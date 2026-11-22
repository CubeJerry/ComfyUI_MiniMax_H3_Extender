"""Run the production orchestrator, archive and cache writers without loading H3."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    return value


class Ref2VAShots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate only external model/runtime imports. Use the actual Extender,
        # Disk Join, disk layout, mode handoff, project code, and both decoders.
        pkg = "extender_shot_test"
        fixtures = {pkg: module(pkg, __path__=[str(ROOT)])}
        for name in ("torch", "torchaudio", "latent_preview", "node_helpers", "folder_paths",
                     "comfy", "comfy.samplers", "comfy.nested_tensor", "comfy.model_management",
                     "comfy.sample", "comfy.sd", "comfy.utils", "comfy.ldm", "comfy.ldm.minimax",
                     "comfy.ldm.minimax.model", "comfy.model_base", "comfy.memory_management"):
            fixtures[name] = module(name, __path__=[])
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(fixtures[parent], child, fixtures[name])
        fixtures["comfy.samplers"].CFGGuider = object
        fixtures["server"] = module("server", PromptServer=types.SimpleNamespace(instance=None))
        for dtype in ("float64", "float32", "float16", "bfloat16", "uint8", "int8", "int16", "int32", "int64", "bool"):
            setattr(fixtures["torch"], dtype, dtype)
        fixtures["torch"].is_tensor = lambda x: False
        fixtures[pkg + ".latent_upscaler"] = module("latent_upscaler", target_dimensions=lambda w,h,s:(w,h),
                                                     upscale_video_latent=Mock())
        cls.addClassCleanup(patch.stopall)
        patch.dict(sys.modules, fixtures).start()
        cls.e = importlib.import_module(pkg + ".extender")
        cls.d = importlib.import_module(pkg + ".motion_context_disk")
        cls.shots = cls.d.shots
        cls.independent = importlib.import_module(pkg + ".ref2va_independent")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stub(self.d, "_CACHE_ROOT", self.root)
        self.e.folder_paths.get_temp_directory = lambda: str(self.root / "preview")
        self.stub(self.e, "_prepare_shared_refs", lambda *a, **k:([], [], [], []))
        self.stub(self.e, "_make_ref2va_conditioning", lambda clip, vae, prompt, *a, **k:(prompt, {}))
        self.stub(self.e, "_release_models_before_refine", lambda *a:None)
        self.stub(self.e, "_resize_context_latent_spatial", lambda proxy, *a:proxy)
        self.stub(self.e, "upscale_for_refine", lambda samples, *a, **k:(samples, 64, 64, None))
        self.stub(self.e, "preserve_first_pass_audio", lambda samples, audio:samples)
        self.stub(self.e, "cache_full_batch_ref2va_segment", Mock())
        self.stub(self.e, "_send_extender_progress", Mock())
        self.samples = []
        self.stub(self.e, "_sample_h3", self.sample)
        self.conditioning = []
        self.stub(self.e.MiniMaxH3MotionContextRAM, "apply", lambda _, *args:self.motion(*args))
        self.stub(self.d, "_write_tensor_raw", self.write_array)
        self.stub(self.e, "_apply_per_clip_loras", lambda *a:(None, None))
        self.cards = [{"id": str(i), "prompt": f"shot {i}", "seed": i, "duration": 2,
                       "validated": False, "incoming_motion_context": i != 4} for i in range(1, 7)]

    def stub(self, obj, name, value):
        p = patch.object(obj, name, value)
        self.addCleanup(p.stop)
        return p.start()

    @staticmethod
    def write_array(stream, array):
        spec = {"shape": list(array.shape), "dtype": "uint8", "offset":stream.tell(), "nbytes":array.nbytes}
        stream.write(array.tobytes())
        return spec

    def sample(self, _model, cond, latent, seed, *args, **kwargs):
        self.samples.append(seed)
        return {"samples": (np.full((1, 1, 17, 4, 4), seed, np.uint8),
                            np.full((1, 1, 2, 94), seed, np.uint8))}

    def motion(self, prompt, latent, previous, context, audio_context):
        parent = previous if isinstance(previous, str) else previous.desc["clip_id"]
        self.conditioning.append((prompt, parent))
        return prompt, int(context), None, 0, None

    def execute(self, mode="full_batch", owner="7", **options):
        state = json.dumps({"clips": self.cards})
        return self.e.MiniMaxH3Extender().extend(
            None, None, None, mode, 64, 64, "match", 6, "lcm", "simple", 1.0,
            "22", 0, state, resolution_mode="manual", unique_id=owner, **options)

    def cache(self, owner="7"):
        return self.e._manifest_for_extender(owner, 24)

    def test_parse_state_and_prompt_pack(self):
        values = [None, 0, "false", False, True]
        for i, flag in enumerate(values):
            self.cards[i]["incoming_motion_context"] = flag
        self.cards[-1].pop("incoming_motion_context")
        self.cards[0]["name"] = "Opening"
        cards = self.e._parse_clips_json(json.dumps(self.cards))
        self.assertEqual([c["incoming_motion_context"] for c in cards], [True, True, True, False, True, True])
        cards, *_ = self.e._sync_clips_from_prompt_pack(cards, {"prompts":["new"]*7})
        restored = self.e._parse_clips_json(self.e._state_json(cards))
        self.assertEqual(restored, cards)
        self.assertEqual(restored[0]["name"], "Opening")
        self.assertEqual(restored[3]["seed"], 4)

    def test_full_batch_refinement_follows_reset(self):
        self.execute(refine_enabled=True)
        self.assertEqual(self.samples, [i for i in range(1, 7) for _ in range(2)])
        self.assertEqual(self.conditioning, [(f"shot {i}", str(i-1)) for i in (2,3,5,6) for _ in range(2)])
        data, _, manifest = self.cache()
        self.assertEqual([s["trim_frames"] for s in manifest["segments"]], [0,22,22,0,22,22])
        self.assertEqual(manifest["final_frame_count"], 6*56-4*22)
        for s in manifest["segments"]:
            for key in ("video", "audio"):
                spec = s[key]
                self.assertEqual(data.read_bytes()[spec["offset"]:spec["offset"]+spec["nbytes"]],
                                 bytes([s["generation_seed"]])*spec["nbytes"])

    def test_clip_by_clip_validation_and_incompatible_cache(self):
        for i in range(6):
            self.execute("clip_by_clip")
            self.cards[i]["validated"] = True
        self.execute("clip_by_clip")
        self.assertEqual(self.samples, list(range(1,7)))
        self.assertEqual(self.conditioning, [(f"shot {i}", str(i-1)) for i in (2,3,5,6)])
        self.cards[3]["incoming_motion_context"] = True
        with self.assertRaisesRegex(ValueError, "Validated Ref2VA clip 4.*incoming motion"):
            self.execute()
        self.cards[3]["incoming_motion_context"] = False
        self.cards[1]["id"] = "retake"
        with self.assertRaisesRegex(ValueError, "cached clip ID"):
            self.execute()
        with self.assertRaisesRegex(ValueError, "disk cache is missing"):
            self.execute(owner="empty")
        self.assertEqual(len(self.samples), 6)

    def interrupt(self):
        with patch.object(self.e, "full_batch_interrupt_requested", side_effect=lambda *a, **k:len(self.samples)==4):
            return self.execute()

    def test_interrupted_archive_resumes_after_reset(self):
        result = self.interrupt()
        self.assertTrue(result["result"][0]["interrupted"])
        payload = result["result"][0]["project_snapshot"]["project"]
        archive = self.root / "mixed.ext"
        self.e._build_project_archive("7", "mixed", payload, archive)
        imported = self.e._import_project_archive("8", archive)
        cards = self.e._clips_from_project_payload(imported["project"])
        self.assertEqual([c["seed"] for c in cards], list(range(1,7)))
        self.assertEqual([c["id"] for c in cards], [c["id"] for c in self.cards])
        self.assertFalse(cards[3]["incoming_motion_context"])
        self.assertEqual(imported["cache"]["computed_indices"], list(range(4)))
        self.cards = cards
        self.execute(owner="8")
        self.assertEqual(self.samples, list(range(1,7)))
        self.assertEqual(self.conditioning[-2:], [("shot 5","4"),("shot 6","5")])

    def test_computed_mismatch_and_parent_retake(self):
        self.interrupt()
        self.cards[2]["incoming_motion_context"] = False
        self.execute()
        self.assertEqual(self.samples, [1,2,3,4,3,4,5,6])
        data, meta, manifest = self.cache()
        for card in self.cards:
            card["validated"] = True
        manifest["segments"][5]["parent_generation_id"] = "wrong"
        self.d._write_json_atomic(meta, manifest)
        with self.assertRaisesRegex(ValueError, "predecessor generation"):
            self.execute()

    def test_clip_zero_can_be_followed_or_skipped(self):
        source = {"id":"a"*64, "width":64,"height":64,"frame_count":56}
        self.stub(self.e, "_continue_video_from_native_input", lambda _:source)
        self.stub(self.e, "_normalize_continue_source", lambda *a:source.copy())
        self.stub(self.e, "_build_continue_context_latent", lambda *a:("source","guide"))
        self.stub(self.e, "_continue_head_guide_conditioning", lambda base,*a:base)
        for incoming in (True, False):
            self.cards[0]["incoming_motion_context"] = incoming
            owner = str(incoming)
            self.execute(owner=owner, continue_existing_video=object())
            _, _, manifest = self.cache(owner)
            first = manifest["segments"][0]
            self.assertEqual(first.get("visible_offset", 0), 22 if incoming else 0)
            self.assertEqual(first["frames"], 34 if incoming else 56)
            self.assertEqual(manifest["source_video"], source)
            self.assertEqual(("shot 1", "source") in self.conditioning, incoming)
            self.conditioning.clear()

    def test_decoder_uses_full_reset_then_pairs_with_it(self):
        self.cards[2]["incoming_motion_context"] = False  # consecutive cuts
        self.cards[-1]["incoming_motion_context"] = False  # final cut
        self.execute()
        data, _, manifest = self.cache()
        segments = manifest["segments"]
        d = self.d
        def read_array(path, spec):
            return np.memmap(path, np.uint8, "r", offset=spec["offset"], shape=tuple(spec["shape"]))
        self.stub(d, "_map_tensor", read_array)
        d.torch.cat = lambda xs, dim: np.concatenate(xs, axis=dim)
        self.stub(d, "_auto_early_seam_shift", lambda *a, **k:0)
        correct = self.stub(d, "_correct_current_segment", Mock(side_effect=lambda _, curr:curr))
        audio_inputs = []
        def decode_audio(_, latent, frames, fps):
            audio_inputs.append(set(np.unique(latent).tolist()))
            return {"waveform":np.arange(frames).reshape(1,1,-1), "sample_rate":24}
        self.stub(d, "_decode_audio_latent", decode_audio)
        video_inputs = []
        def decode_video(latent):
            video_inputs.append(set(np.unique(latent).tolist()))
            return np.arange(d._frames_from_video_t(latent.shape[2])).reshape(-1,1,1,1)
        vae = types.SimpleNamespace(decode=decode_video)
        for trim in (5, 22, 39):
            audio_inputs.clear()
            video_inputs.clear()
            correct.reset_mock()
            for i, desc in enumerate(segments):
                if i in (1, 4):
                    desc["trim_frames"] = trim
                video, audio, shift = d._render_one_final_segment(data, segments, i, vae, None, 24)
                wanted = 56 - desc["trim_frames"]
                self.assertEqual(len(video), wanted)
                self.assertEqual(audio["waveform"].shape[-1], wanted)
                self.assertEqual(shift, 0)
            self.assertEqual(correct.call_count, 2)
            self.assertEqual(video_inputs, [{1},{1,2},{3},{4},{4,5},{6}])
            self.assertEqual(audio_inputs, video_inputs)
        cropped = dict(segments[3], frames=48, source_frames=56, visible_offset=8)
        video, audio, _ = d._render_one_final_segment(data, [segments[0], cropped], 1, vae, None, 24)
        self.assertEqual(video[:,0,0,0].tolist(), list(range(8,56)))
        self.assertEqual(audio["waveform"].ravel().tolist(), list(range(8,56)))

    def test_invalid_descriptors_and_legacy_zero(self):
        good = {"frames":56,"trim_frames":0}
        self.assertFalse(self.shots.recorded_motion(good, 1))
        for changes in ({"trim_frames":None}, {"trim_frames":-1}, {"trim_frames":"0"},
                        {"trim_frames":True}, {"trim_frames":6}, {"incoming_motion_context": True},
                        {"incoming_motion_context": False,"trim_frames":22},
                        {"incoming_motion_context":"false"}, {"visible_offset":10}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.shots.recorded_motion(dict(good, **changes), 1)
        with self.assertRaises(ValueError):
            self.shots.recorded_motion({"frames":56}, 1)
        with self.assertRaises(ValueError):
            self.shots.restore_causal_geometry(dict(good, source_frames=56, source_trim_frames=22, visible_offset=22), 1)

    def test_legacy_derived_migration_is_selective_and_idempotent(self):
        self.execute()
        data, meta, manifest = self.cache()
        original = data.read_bytes()
        for s in manifest["segments"]:
            s.pop("shot_decode_version", None)
            s.pop("incoming_motion_context", None)
            s["decoded_audio"] = {}
            s["decoded_mp4_blob"] = {}
            s["final_video_cache_version"] = 1
        manifest.update(preview_committed_count=6)
        self.d._write_json_atomic(meta, manifest)
        updated = self.d._load_manifest_from_paths(data, meta)
        self.assertNotIn("preview_committed_count", updated)
        self.assertEqual(["decoded_audio" in s for s in updated["segments"]], [True]*3+[False]*3)
        self.assertEqual(["decoded_mp4_blob" in s for s in updated["segments"]], [True]*3+[False]+[True]*2)
        self.assertNotIn("final_video_cache_version", updated["segments"][3])
        self.assertFalse(self.shots.migrate_decoded_shots(updated))
        self.assertEqual(data.read_bytes(), original)

    def test_gain_and_individual_pcm_restart_at_cuts(self):
        self.execute()
        data, meta, manifest = self.cache()
        for s in manifest["segments"]:
            s["decoded_audio"] = {}
        def cached(path, desc):
            n = desc["frames"] - desc["trim_frames"]
            return {"waveform":np.full((1,1,n), desc["index"]+1), "sample_rate":24}
        self.stub(self.d, "_load_cached_decoded_audio", cached)
        pair = self.stub(self.d, "_decode_pair_audio", Mock(return_value=(cached(None, manifest["segments"][0]), 56, 34)))
        self.stub(self.d, "_match_pair_gain_to_previous", lambda *a:2)
        updated = self.d._upgrade_cached_audio_gain_chain(data, meta, manifest, None, 24)
        self.assertEqual([s["decoded_audio"]["timeline_gain"] for s in updated["segments"]], [1,2,2,1,2,2])
        self.assertEqual([call.args[2]["clip_id"] for call in pair.call_args_list], ["2","3","5","6"])
        smooth = self.stub(self.d, "_smooth_segment_entry_level", Mock(side_effect=lambda prev,w,sr:w))
        self.stub(self.d, "_declick_segment", lambda prev,w,*a:w)
        self.stub(self.d, "_audio_seam_tail", lambda wave,sr:wave[..., -1:])
        self.stub(self.d, "_write_audio_raw", lambda stream,w:stream.write(w.astype("uint8").tobytes()))
        paths = [self.root/f"{i}.pcm" for i in range(6)]
        total = self.root/"all.pcm"
        _, _, n = self.d._write_preview_pcm_audio(None,data,updated["segments"],6,24,total,"test",paths)
        self.assertEqual([int(c.args[1][0,0,0]) for c in smooth.call_args_list], [2,3,5,6])
        self.assertEqual(total.read_bytes(), b"".join(p.read_bytes() for p in paths))
        self.assertEqual(n, 6*56 - 4*22)

    def test_roundtrip_including_clip_zero_and_stale_target(self):
        self.cards = self.cards[:4]
        self.execute()
        data, meta, before = self.cache()
        first = before["segments"][0]
        first.update(continued_from_source=True, source_frames=56, visible_offset=22, frames=34,
                     parent_generation_id="source")
        before["source_video"] = {"id":"source", "width":64,"height":64,"frame_count":100}
        before["final_frame_count"] = self.d._final_frame_count(before["segments"])
        self.d._write_json_atomic(meta, before)
        self.d._source_working_path(data).write_bytes(b"source video")
        for c in self.cards:
            c["validated"] = True
        saved_bytes = data.read_bytes()
        off = self.d._handoff_ref2va_cache("7", True, self.cards)
        self.assertEqual(off["validated_ids"], ["1","2","3","4"])
        self.assertEqual(data.read_bytes(), saved_bytes)
        other_data, other_meta = self.d._chain_paths(self.independent.cache_owner_id("7"))
        independent = self.d._load_manifest_from_paths(other_data, other_meta)
        self.assertEqual(independent["segments"][0]["visible_offset"], 22)
        self.assertEqual(independent["segments"][1]["frames"], 34)
        # Same id, same length, new take; source cache was not changed.
        samples = self.sample(None,None,None,44)
        self.independent.store_segment("7", 24, ["1","2","3","4"], 3, "4", samples,
                                       validated=True, generation_seed=44)
        other_bytes = other_data.read_bytes()
        result = self.d._handoff_ref2va_cache("7", False, self.cards)
        self.assertEqual(other_data.read_bytes(), other_bytes)
        self.assertEqual(result["incoming_updates"], {"4":False})
        after = self.cache()[2]
        self.assertEqual([s["trim_frames"] for s in after["segments"]], [0,22,22,0])
        self.assertEqual(after["segments"][0]["visible_offset"], 22)
        self.assertEqual(after["segments"][0]["source_frames"], 56)
        self.assertEqual(after["final_frame_count"], before["final_frame_count"])
        self.assertEqual(after["segments"][-1]["generation_seed"], 44)
        self.assertEqual(after["source_video"], before["source_video"])
        self.assertEqual(self.d._source_working_path(data).read_bytes(), b"source video")
        self.assertEqual(data.read_bytes()[:before["segments"][2]["segment_end"]],
                         saved_bytes[:before["segments"][2]["segment_end"]])
        self.assertEqual([s["clip_id"] for s in after["segments"]], ["1","2","3","4"])
        self.assertTrue(all(s["validated"] for s in after["segments"]))

    def test_ambiguous_and_failed_handoffs_preserve_branches(self):
        self.execute()
        self.d._handoff_ref2va_cache("7", True, self.cards)
        data, meta, manifest = self.cache()
        other_data, other_meta = self.d._chain_paths(self.independent.cache_owner_id("7"))
        off = self.d._load_manifest_from_paths(other_data, other_meta)
        off["segments"][-1]["generation_id"] = "off-retake"
        self.d._write_json_atomic(other_meta, off)
        before = [p.read_bytes() for p in (data, meta, other_data, other_meta)]
        # A failed install must restore the preexisting target even if the
        # exception occurs midway through backups and installs.
        replace = self.e.os.replace
        def fail(staged, target):
            if Path(target)==meta and Path(staged).name=="chain.json":
                raise OSError("injected failure")
            return replace(staged, target)
        with patch.object(self.e.os, "replace", side_effect=fail), self.assertRaises(OSError):
            self.d._handoff_ref2va_cache("7", False, self.cards)
        self.assertEqual([p.read_bytes() for p in (data,meta,other_data,other_meta)], before)
        manifest["segments"][-1]["generation_id"] = "causal-retake"
        self.d._write_json_atomic(meta, manifest)
        before[1] = meta.read_bytes()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.d._handoff_ref2va_cache("7", False, self.cards)
        self.assertEqual([p.read_bytes() for p in (data,meta,other_data,other_meta)], before)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg/ffprobe")
    def test_ffmpeg_final_and_individual_lengths(self):
        self.execute()
        data, _, manifest = self.cache()
        segments = manifest["segments"]
        counts = [d["frames"]-d["trim_frames"] for d in segments]
        profile = {"codec":"FFV1 lossless", "crf":17, "preset":"fast"}
        paths = []
        for i, n in enumerate(counts):
            path = self.root / f"segment-{i}.mkv"
            subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=size=16x16:rate=24",
                            "-frames:v",str(n),"-c:v","ffv1",str(path)], check=True)
            paths.append(path)
        self.stub(self.d, "_load_cached_decoded_audio", lambda _, desc: {
            "sample_rate":24000, "waveform":np.zeros((1,2,counts[desc["index"]]*1000), np.float32)})
        self.stub(self.d, "_smooth_segment_entry_level", lambda _,w,sr:w)
        self.stub(self.d, "_declick_segment", lambda _,w,*a:w)
        self.stub(self.d, "_audio_seam_tail", lambda w,sr:w[...,-1:])
        self.stub(self.d, "_write_audio_raw", lambda f,w:f.write(w[0].T.copy().tobytes()))
        final = self.root / "final.mkv"
        _, result = self.d._export_final_from_exact_segment_caches(
            "ffmpeg", paths, data, segments, 24, final, profile, "192k", "export",
            save_individual_clips=True, manifest=manifest)
        self.assertNotIn("individual_clips_error",result)
        files = sorted(Path(result["individual_clips_dir"]).glob("*.mkv"))
        self.assertEqual(len(files),len(segments))
        for path, expected in zip([final]+files, [sum(counts)]+counts):
            frames = subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)])
            pcm = subprocess.check_output(["ffmpeg","-v","error","-i",str(path),"-map","0:a:0",
                                           "-f","s16le","-"])
            self.assertEqual(int(frames), expected)
            self.assertEqual(len(pcm), expected*1000*2*2)

    def test_autosave_retains_motion_and_actual_sampler_seed(self):
        self.execute()
        _, _, manifest = self.cache()
        for c in self.cards:
            c["validated"] = True
            c["seed"] += 20  # next queued seed can differ from generated seed
        unchanged = self.execute()
        self.assertEqual(self.samples,list(range(1,7)))
        video = self.root / "finished.mp4"
        video.touch()
        self.e._auto_save_full_batch_project(unchanged["result"][0], video, "8",
                                              {"auto_save_project":True}, 6, manifest["final_frame_count"])
        imported = self.e._import_project_archive("8", video.with_suffix(".ext"))
        restored = self.e._clips_from_project_payload(imported["project"])
        self.assertEqual([c["seed"] for c in restored], list(range(1,7)))
        self.assertEqual([c["incoming_motion_context"] for c in restored], [True,True,True,False,True,True])
