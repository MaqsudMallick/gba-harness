# neatHack remaining actions

This checklist contains only work that still requires an account owner, browser session, or
submission asset. Code, secrets, private account data, and payment information do not belong
in this file.

## Product milestones

- [x] Run 2-4 GBA instances in one linked mGBA core with independent controllers.
- [x] Apply multiple agents' inputs atomically on a shared emulated clock.
- [x] Split the composite framebuffer into native per-player visual observations.
- [x] Verify Mario Kart: Super Circuit reaches a genuine `2P` linked lobby.
- [x] Add a modular Player 2 visual steering/throttle agent, automatic quick-race setup,
      live agent/input status, and off-road/reverse recovery for Mario Kart.
- [x] Complete a verified three-lap Mario Kart race using the local visual controller.
      Deterministic headless evaluation finished Peach Circuit in `5:46.75` with lap
      times `2:22.21`, `1:56.77`, and `1:27.77`.
- [ ] Add match telemetry: per-player decisions, recovery, lap/progress, finish order, and
      a Neatlogs trace that makes the competition inspectable.
- [ ] Add gamepad bindings for human P3/P4; the backend already exposes four controller
      ports, while the current keyboard demo maps P1 and P2.

## Entire checkpoint

- [x] Refreshed the Entire login and connected the repository.
- [x] Confirmed `/gh/MaqsudMallick/gba-harness` is ready and mirrored.
- [x] Attached the Codex session to commit `c47ae93c9583137756b38c9a89c1b5cba36f04c7`
      and verified checkpoint `01M4K4DYFVKB5TF5R77KZMJD5J` locally with
      `entire checkpoint list` and `entire checkpoint explain HEAD`.
- [x] Opened the repository in Entire and confirmed the recorded checkpoint target:
      <https://entire.io/gh/MaqsudMallick/gba-harness/commit/c47ae93c9583137756b38c9a89c1b5cba36f04c7>.

Recorded GitHub-backed checkpoint URL:

```text
https://entire.io/gh/MaqsudMallick/gba-harness/commit/c47ae93c9583137756b38c9a89c1b5cba36f04c7
```

## Neatlogs evidence

- [ ] In the Neatlogs dashboard, open the application traces recorded in `hackathon.md`.
- [ ] Save sanitized shareable trace links or screenshots for the before/failure-recovery and
      after/fix runs. Do not publish API keys, account details, raw ROM data, or private inputs.
- [ ] Confirm the hackathon plan entitlement is active for the project.

The instrumented runner now prints `neatlogs_trace_id` after every run, including failures,
so future traces can be located without guessing.

Completed level-clear application trace: `857e03574e75b220897fd4c0c42dc5b1` (126 turns,
2,928 emulated frames, status `completed`, provider `local-vision`).

## cfo.ai business plan

- [ ] Sign up through <https://cfo.ai/neathacks> using the team leader's account.
- [ ] Complete the card-required signup, then immediately cancel in **Settings -> Billing** as
      instructed by the organisers.
- [ ] Ask the organisers to confirm the neatHack credits have been applied.
- [ ] Use `cfo-ai-brief.md` as the input and build conservative, base, and growth scenarios
      covering pricing, costs, revenue, gross margin, burn, runway, and break-even.
- [ ] Replace assumptions with measured trace/model costs where available.
- [ ] Export or share the resulting cfo.ai business plan and add its link or sanitized artifact
      to `hackathon.md` and the final submission.

## Crustdata credits

- [ ] Open <https://app.crustdata.com> and select **Have a promo code?** before signup.
- [ ] Apply promo code `NEATLOGS-OCT2026`.
- [ ] Sign up with email, not **Continue with Google**, so the promo code is retained.
- [ ] Confirm the account received 5,000 credits. Contact the organisers in Discord if the
      account already exists or the credits do not appear.

## Community and submission

- [ ] Join the neatHack Discord through <https://neatlogs.com/discord> and monitor it for the
      final submission cutoff and organiser announcements.
- [ ] Ensure the repository is public and the pushed hackathon commits are visible to judges.
- [ ] Record a demo video of three minutes or less covering architecture, implementation,
      runtime, reasoning, and results for neatlogs, Entire, and cfo.ai.
- [ ] Include the public repository, demo, Neatlogs evidence, Entire checkpoint URL and ID,
      cfo.ai business plan, and public `#neatHack` build links in the submission.
- [ ] Submit during the announced window by quoting the official neatlogs submission post;
      confirm the final cutoff in Discord before publishing.

## Optional Entire features

- [ ] If useful, request Trails access and configure its runners only after access is granted.
- [ ] If useful, evaluate Entire Graph/Brain after the required checkpoint is accessible.

