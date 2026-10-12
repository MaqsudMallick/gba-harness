"""Run Zelda and capture exact model and controller data for the local dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "zelda-live-trace"
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--max-turns", type=int, default=40)
    result.add_argument("--max-model-calls", type=int, default=40)
    return result


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from gbharness.agent.linked_runner import LinkedAgentRunner
    from gbharness.agent.vertex_team import VertexTeamDecisionProvider
    from gbharness.vision.encode import to_png

    args = parser().parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started_at = time.time()
    events_path = output / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    def append_event(event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "sequence": len(calls) + len(executions),
            "time_unix": time.time(),
            "type": event_type,
            **payload,
        }
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save_snapshot() -> None:
        payload = {
            "capture": {
                "started_unix": started_at,
                "updated_unix": time.time(),
                "call_count": len(calls),
                "execution_count": len(executions),
            },
            "calls": calls,
            "executions": executions,
            "errors": errors,
        }
        temporary = output / "trace.json.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(output / "trace.json")

    original_decide_team = VertexTeamDecisionProvider.decide_team
    original_execute_plans = LinkedAgentRunner._execute_plans
    original_record = LinkedAgentRunner._record

    def capture_decide_team(self: VertexTeamDecisionProvider, request: Any) -> Any:
        call_number = len(calls) + 1
        started = time.perf_counter()
        try:
            decision = original_decide_team(self, request)
        except BaseException as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            error = {
                "call": call_number,
                "turn": request.turn,
                "stage_before": request.stage,
                "latency_ms": latency_ms,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            errors.append(error)
            append_event("model_call_failed", error)
            save_snapshot()
            print(
                f"FAILED_MODEL_CALL={call_number} TURN={request.turn} "
                f"TYPE={type(exc).__name__} MESSAGE={exc}",
                flush=True,
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        trace_input = json.loads(decision.trace_input_json)
        expected_hash = trace_input["observation"]["sha256"]
        candidates = (request.image_png, *request.player_images)
        observation = next(
            image for image in candidates if hashlib.sha256(image).hexdigest() == expected_hash
        )
        input_name = f"call-{call_number:03d}-input.png"
        (output / input_name).write_bytes(observation)

        raw = next(iter(decision.decisions.values())).raw
        call = {
            "call": call_number,
            "turn": request.turn,
            "stage_before": request.stage,
            "latency_ms": latency_ms,
            "input_image": input_name,
            "input_image_sha256": expected_hash,
            "input": trace_input,
            "raw_model_response_text": getattr(raw, "text", None),
            "output": json.loads(decision.trace_output_json),
        }
        calls.append(call)
        append_event("model_call_completed", call)
        save_snapshot()
        print(
            f"CAPTURED_MODEL_CALL={call_number} TURN={request.turn} "
            f"STAGE={request.stage} LATENCY_MS={latency_ms:.1f}",
            flush=True,
        )
        return decision

    def capture_execute_plans(self: LinkedAgentRunner, plans: Any) -> Any:
        execution_number = len(executions) + 1
        before_png = to_png(self.frames.require())
        before_name = f"execution-{execution_number:03d}-before.png"
        (output / before_name).write_bytes(before_png)
        result, executed_plans = original_execute_plans(self, plans)
        after_png = to_png(self.frames.require())
        after_name = f"execution-{execution_number:03d}-after.png"
        (output / after_name).write_bytes(after_png)
        execution = {
            "execution": execution_number,
            "requested_plans": [list(plan) for plan in plans],
            "executed_plans": [list(plan) for plan in executed_plans],
            "frame_after": result.total_frames,
            "frames_advanced": result.frames_advanced,
            "emulation_ended": result.emulation_ended,
            "before_image": before_name,
            "before_image_sha256": hashlib.sha256(before_png).hexdigest(),
            "after_image": after_name,
            "after_image_sha256": hashlib.sha256(after_png).hexdigest(),
        }
        executions.append(execution)
        append_event("controller_execution_completed", execution)
        save_snapshot()
        return result, executed_plans

    def capture_record(self: LinkedAgentRunner, *record_args: Any, **record_kwargs: Any) -> Any:
        record = original_record(self, *record_args, **record_kwargs)
        if executions:
            execution = executions[-1]
            execution["runtime_stage_after"] = record.team_stage
            execution["progress_detected"] = record.progress_detected
            execution["changed_players"] = list(record.changed_players)
            execution["execution_steps"] = record.execution_steps
            append_event(
                "turn_finalized",
                {
                    "turn": record.turn,
                    "runtime_stage_after": record.team_stage,
                    "progress_detected": record.progress_detected,
                    "changed_players": list(record.changed_players),
                },
            )
            save_snapshot()
        return record

    VertexTeamDecisionProvider.decide_team = capture_decide_team
    LinkedAgentRunner._execute_plans = capture_execute_plans
    LinkedAgentRunner._record = capture_record
    save_snapshot()

    from examples.zelda_four_swords import main as run_zelda

    return run_zelda(
        [
            "--live",
            "--skip-first-puzzle",
            "--npc-autopilot",
            "--max-turns",
            str(args.max_turns),
            "--max-model-calls",
            str(args.max_model_calls),
            "--log-turns",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
