# Astra handoff: per-clip motion resets and clip navigation

Implement the changes below in [CubeJerry/ComfyUI_MiniMax_H3_Extender](https://github.com/CubeJerry/ComfyUI_MiniMax_H3_Extender). Finish with tested source changes and a commit or pull request in this fork. Inspect the current branch before editing and preserve unrelated work.

The goal is to mix continuous shots and independent cuts in one Ref2VA sequence. A clip must be able to ignore incoming motion while still supplying its generated motion to the next clip. This must work in both `clip_by_clip` and `full_batch`, including interrupted batch resumes and existing projects.

## Starting point

The reviewed upstream baseline is commit `009088b66b805adf099555b3c73d51b66eef4f2f`, version 3.1.1. At that commit, the fork has no per-clip incoming-motion field, reset-aware final decoder, or complete OFF-to-ON cache handoff. Local repair scripts were used outside the repository. Implement the behavior directly in source; do not make users run string-replacement patch installers.

This preparation adds a frontend fix: `render()` captures the clip strip's scroll position before rebuilding its contents, then restores it synchronously with instant scrolling. It also preserves the reference strip's position and FL2VA alignment. Keep this fix and its tests. The Node suite passed 27 tests, including DOM fixtures that call the production renderer. Live ComfyUI browser checks and GPU generation remain unverified. The fixtures do not establish compatibility with either live Legacy or Nodes 2.0 rendering.

Read these areas first:

- `extender.py`: clip defaults/parsing, project serialization/import, the shared Ref2VA loop, refinement, and batch checkpoints.
- `motion_context_disk.py`: DiskJoin, cache bootstrap, segment renderers, audio gain migration, preview repair, and Full Batch caches.
- `motion_context_ram.py`: context-tail extraction and H3 temporal geometry.
- `ref2va_independent.py`: global Motion OFF behavior and independent descriptors.
- `web/extender.js`: normalization, serialization, validation, mode switching, cache-state responses, and `render()`.
- Existing tests for project restoration, batch seeds, future local-reference edits, prompt scrolling, and clip-strip scrolling.

## Required motion behavior

Keep global **Motion Context ON** for sequences containing continuations and cuts. Add a per-clip boolean, preferably `incoming_motion_context`, meaning whether this clip uses motion from its immediate predecessor. Default missing settings to ON for existing workflows.

| Clip | Incoming motion | Generation source | Output available to the next clip |
| --- | --- | --- | --- |
| 1 | ON by default | No preceding generated clip | Clip 1's full latent |
| 2 | ON | Clip 1 | Clip 2's full latent |
| 3 | ON | Clip 2 | Clip 3's full latent |
| 4 | OFF | Independent generation | Clip 4's full latent |
| 5 | ON | Clip 4 | Clip 5's full latent |
| 6 | ON | Clip 5 | Clip 6's full latent |

Clip 4's switch changes incoming conditioning only. Write its complete video/audio latent to the same sequential cache and advance the normal handle and proxy. Clip 5 must consume clip 4, without returning to clip 3 or another cache owner's older timeline. Support consecutive resets and a reset on the last clip.

Keep global Motion OFF's existing independent editing behavior. Explain the two controls clearly. A card label such as “Use motion from previous clip” should explain that OFF starts a new shot whose output can still guide the following clip.

### State and generation

Normalize the field consistently in Python and JavaScript. Python `bool("false")` enables motion; JavaScript `value !== false` accepts malformed values differently. Accept actual booleans, provide a documented legacy default, and reject or consistently normalize other values.

Preserve the field through defaults, copied clips, frontend normalization, hidden widgets, workflow JSON, mode snapshots, prompt-pack updates, backend responses, `.ext` save/load, and autosave. `_parse_clips_json` reconstructs dictionaries and currently drops unknown fields; a JSON widget alone is insufficient. Preserve IDs, prompts, generation seeds, reference slots, names, and validation state.

Gate `motion.apply` in the shared Ref2VA loop. Gate `previous_refined_proxy` in the refinement pass too. Pass zero overlap trim explicitly for a later independent clip; DiskJoin's default continuation trim must not apply. Always store the generated result and update the predecessor proxy afterward.

Define the first generated clip's control when a continued source video, Clip 0, exists. Without Clip 0, explain or disable the irrelevant incoming control. With Clip 0, expose whether the first generated clip follows that source or starts independently. Preserve the source video's inclusion and frame accounting.

### Reuse, validation, and Full Batch

Check compatibility before both reuse paths: validated clips and unvalidated `computed` Full Batch checkpoints. The earlier repair script checked only validated reuse. Verify incoming-motion metadata and clip identity/lineage before accepting a cached result.

A matching validated or computed clip must not be sampled again. Missing or incompatible validated cache must produce a specific error, without silently clearing validation or regenerating accepted work. A changed incoming setting must make affected computed checkpoints ineligible for reuse. Preserve valid checkpoints after interruption, save/load, and restart, including their actual generation seeds.

Use conservative causal invalidation: changing a clip's incoming setting invalidates that clip and following clips. Keep the accepted prefix. Do not add reset-boundary dependency optimization unless its correctness is separately established.

Allow switches on future, ungenerated cards. Use the existing physical-cache guard before requesting disk invalidation. The current backend tolerates unvalidation beyond the cached prefix, so do not describe that case as a confirmed server error. On persistence failure, restore consistent settings and validation/computed state. Ignore stale asynchronous cache responses after mode or project changes.

Validation commits an existing cached result; it must not initiate sampling. Distinguish desired card settings from those recorded when a cached clip was generated. Do not relabel an old latent to make it appear compatible. Do not promise that queuing an entire workflow performs decode only: later unvalidated cards can still generate.

## Decode independent cuts correctly

Preserve H3's temporal phase checks. Valid video latents use `T = 2 mod 5`; concatenating independent streams at zero trim breaks the paired decoder. Do not remove the guard, invent overlap, or alter latent payloads to suppress the error.

Decode reset video and audio standalone. Preserve the visible-frame window and skip predecessor-based seam/color/gain matching at that cut. The following ON segment must use ordinary paired decode with the reset clip as predecessor. Apply these rules to progressive preview, missing derived-cache repair, Full Batch segment caching, final assembly, and individual clip export.

Review `_render_one_final_video_segment`, `_render_one_final_audio_segment`, and `_upgrade_cached_audio_gain_chain`. Audio gain migration must not pair-decode a reset or carry predecessor gain across the cut. Check derived-cache metadata so stale decoded artifacts cannot bypass corrected behavior.

Validate descriptors before selecting a decode path. The repair helper treated missing trim as zero and therefore as a reset. It rejected explicit OFF with positive trim, but did not reject a later explicit ON clip with zero trim. Define valid first-clip and Clip 0 exceptions, then reject missing, malformed, negative, or contradictory geometry where it cannot be recovered safely. Infer legacy reset status from restored geometry only when the descriptor supports it. Keep genuine legacy zero-overlap clips usable.

## Preserve caches during migration

The global toggle uses separate causal and independent cache owners. The baseline can return an older causal cache on OFF-to-ON because an existing target short-circuits the handoff. ON-to-OFF also records source frame/trim metadata that must be restored when returning to causal representation.

Repair this path if global switching remains supported. Restore causal `frames` and `trim_frames` from `source_frames` and `source_trim_frames`, checking that the saved visible length agrees. Preserve newly independent clips with zero overlap. Verify visible offsets, source-video metadata, clip order, and validation after a round trip.

Do not choose between conflicting retakes using only cache length, clip IDs, modification time, or an independent segment's presence. Equal-length branches can contain different generations under the same IDs. Use trustworthy lineage/generation metadata; otherwise report ambiguity and preserve both caches. Apply a selected handoff transactionally. Change frontend mode and validation snapshots only after success. Responses for the old mode must not overwrite the new state.

Preserve raw generated latent payloads and projects throughout decode repair and metadata migration. Version migrations where needed, make them idempotent, and document recovery. Restrict derived-cache rebuilding to what the corrected decoder requires. Keep cache versions and build identifiers coherent with the implemented schema.

## Many-clip navigation

The preparation fixes the immediate horizontal jump caused by rebuilding the card strip. Maintain this behavior when adding motion controls. Preserve each prompt's caret/selection and internal scroll, and respect the existing renderer-specific focus behavior. Refreshes must not steal focus. Preserve both axes where applicable and FL2VA media alignment.

Test motion changes, validation, local-reference changes, and asynchronous cache refreshes while scrolled to a later card. Include two Extender nodes so one node's render cannot restore another's position. Cover clip removal, resizing, and deliberate “Add clip” navigation without overriding the intended destination. Manually verify Legacy and Nodes 2.0 frontends where available.

If navigation needs more work, prefer a compact jump-to-clip control showing clip number and optional name, plus visible reset markers. Keep existing cards and editing behavior. A large redesign is optional; do not make it a prerequisite for motion support. Verify keyboard access and many clips in a live ComfyUI session.

## Editable Ref2VA prompt

The user also reported an uneditable prompt in a separate Pika Ref2VA workflow. That supplied workflow is not one of the fork's three shipped examples. Inspect its wiring before changing shared node behavior.

The earlier workaround moved text from a linked `PrimitiveStringMultiline` into the `MiniMaxH3ReferenceToVideo` prompt widget and removed the obsolete node/link. A connected input may intentionally make the receiving widget non-editable; separately check browser focus and pointer handling. Provide one obvious editable source, with typing, multiline paste, selection, save/reload, and generation all using the same value.

Preserve the original text locally when transforming the user's workflow. Publish only a portable example with a neutral prompt and sanitized paths/state. Do not commit private prompts, generated caches, project archives, or machine-specific paths. If the original file is unavailable, state the limitation and provide a neutral reproduction instead of claiming verification.

## Verification and delivery

Add CPU regression tests exercising production parsing, orchestration, cache compatibility, and decoder selection, with controlled stubs where GPU objects are required. Do not test a duplicate implementation or rely only on string matches.

| Area | Required checks |
| --- | --- |
| State | Boolean normalization; legacy defaults; workflow and `.ext` round trips; autosave; stable IDs and seeds |
| Generation | Mixed ON/OFF in both modes; reset cached; successor reads reset; refinement; Clip 0 |
| Reuse | Validated and computed skips; missing cache; changed flags; wrong lineage; interrupted batch resume |
| Decode | Standalone reset AV; paired successor; consecutive/final resets; context lengths; visible windows; invalid metadata |
| Migration | ON/OFF round trips; stale target; ambiguous retakes; failed handoff; unchanged source latent bytes |
| UI | Future cards; failed persistence; later-card scrolling; two nodes; prompt focus; editable workflow wiring |

Run the existing suites, including `node --test tests/*.test.mjs` and `python -m unittest discover -s tests -p 'test_*.py'`, adapting the environment only as needed. Report failures with their cause.

Then perform a live ComfyUI smoke test: global ON, clip 4 OFF, clip 5 ON, in both run modes. Confirm clip 5's source, video/audio decode, lengths, and absence of unwanted sampling. Interrupt Full Batch after a reset, restart or reload its project, and verify continuation from the correct cached predecessor. Repeat with refinement if supported. If GPU or browser access is unavailable, leave those checks explicitly pending.

Keep the short fork section at the top of README factual. Separate shipped changes from planned work until implementation and verification justify updating it. Retain upstream attribution and documentation. Use the no-ai-slop skill when available to edit the wording. Use plain language without promotional claims.

Deliver the commit or PR, explain changes, list tests actually run, and identify pending checks. Include upgrade instructions for local-hotfix users: preserve their source changes before updating, avoid blindly applying old scripts again, restart ComfyUI, and refresh frontend assets. Preserve existing clips and projects while moving to the integrated implementation.
