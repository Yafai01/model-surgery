import subprocess
import sys
import time
from pathlib import Path

import pytest
from helpers import hf_llama_dir, llama_like_gguf

import model_surgery.jobs as jobs_mod
from model_surgery.config import Settings
from model_surgery.history import History
from model_surgery.jobs import ExecuteRequest, JobManager, PlanRequest
from model_surgery.scanner import scan_path
from model_surgery.tooling import ConvertTool, QuantizeTool, QuantType, ToolsReport

FAKE = Path(__file__).with_name("fake_llama_quantize.py")


@pytest.fixture
def env(tmp_path, monkeypatch):
    src_dir = tmp_path / "models"
    src_dir.mkdir()
    src = llama_like_gguf(src_dir / "tiny.gguf", wtype="F16", ftype=1)
    expected = llama_like_gguf(tmp_path / "expected_q4_0.gguf", wtype="Q4_0", ftype=2)
    fake_exe = tmp_path / "bin" / "llama-quantize.exe"
    fake_exe.parent.mkdir()
    fake_exe.write_bytes(b"")

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    history = History(settings.db_path)
    report = ToolsReport(
        quantize=QuantizeTool(
            available=True,
            path=str(fake_exe),
            version="test",
            types=[QuantType(name=n, ftype_id=i, description="") for n, i in [("Q4_0", 2), ("Q8_0", 7), ("IQ2_XS", 20), ("Q4_K_M", 15)]],
        ),
        convert=ConvertTool(available=False, detail="not configured"),
        checked_at=0,
        configuration_help=[],
    )
    mgr = JobManager(settings, history, tools_fn=lambda refresh=False: report)

    real_popen = subprocess.Popen

    def popen(argv, *a, **kw):
        if argv and argv[0] == str(fake_exe):
            argv = [sys.executable, str(FAKE)] + list(argv[1:])
        return real_popen(argv, *a, **kw)

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", popen)
    monkeypatch.setenv("FAKE_OUTPUT_SRC", str(expected))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_DELAY", raising=False)
    scan = scan_path(str(src), settings)
    history.save_scan(scan)

    class E:
        pass

    e = E()
    e.mgr, e.scan, e.settings, e.history, e.report, e.tmp, e.src = mgr, scan, settings, history, report, tmp_path, src
    return e


def wait(mgr, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = mgr.get(job_id)
        if j.status not in ("running", "validating"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def run(env, target="Q4_0", **kw):
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target=target, **kw), env.scan)
    assert plan.can_execute, plan.blockers
    job = env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=True, confirm_token=plan.confirm_token))
    return plan, wait(env.mgr, job.id)


def test_plan_is_dry_run_only(env):
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0"), env.scan)
    assert plan.can_execute and plan.confirm_token
    assert plan.argv[0].endswith("llama-quantize.exe") and plan.argv[-1] == "Q4_0"
    assert plan.argv[-2].endswith(".Q4_0.gguf.partial")
    assert plan.estimated_output_bytes and plan.free_disk_bytes
    assert not Path(plan.output_dir).joinpath("tiny.Q4_0.gguf").exists()
    assert Path(plan.output_dir).resolve() != env.src.parent.resolve()


def test_successful_job_is_validated(env):
    plan, job = run(env)
    assert job.status == "succeeded", job.error
    assert job.validation and job.validation.passed
    assert {c.name for c in job.validation.checks} >= {"exit status", "GGUF readable", "tensor set preserved", "file type"}
    out = Path(job.output_path)
    assert out.is_file() and out.name == "tiny.Q4_0.gguf"
    assert not Path(plan.partial_path).exists()
    assert job.output_size_bytes == out.stat().st_size
    assert job.progress == 1.0 and job.output_scan_id
    assert env.src.read_bytes()[:4] == b"GGUF"  # source untouched
    # history persisted
    assert env.history.get_job(job.id)["status"] == "succeeded"


def test_tool_failure(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    plan, job = run(env)
    assert job.status == "failed" and job.exit_code == 1
    assert "simulated failure" in job.error
    assert not Path(plan.partial_path).exists() and not Path(plan.output_path).exists()


def test_unreadable_output_fails_validation(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "garbage")
    plan, job = run(env)
    assert job.status == "failed" and "validation" in job.error
    assert not next(c for c in job.validation.checks if c.name == "GGUF readable").passed
    assert not Path(plan.output_path).exists()


def test_wrong_file_type_fails_validation(env, monkeypatch):
    monkeypatch.setenv("FAKE_OUTPUT_SRC", str(env.src))  # tool "succeeds" but writes an F16 file
    _, job = run(env)
    assert job.status == "failed"
    assert not next(c for c in job.validation.checks if c.name == "file type").passed


def test_no_overwrite(env):
    first, _ = run(env)
    # explicit directory that already holds the output: blocked
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0", output_dir=first.output_dir), env.scan)
    assert not plan.can_execute and any("never overwrites" in b for b in plan.blockers)
    # default directory: a fresh sibling folder is chosen instead
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0"), env.scan)
    assert plan.can_execute and plan.output_dir != first.output_dir and plan.output_dir.endswith("-2")


def test_default_output_never_collides_with_source(env):
    # quantizing a model that itself lives in the default output folder must not be blocked
    _, job = run(env)
    out_scan = scan_path(job.output_path, env.settings)
    plan = env.mgr.create_plan(PlanRequest(scan_id=out_scan.id, operation="quantize", target="Q8_0", allow_requantize=True), out_scan)
    assert plan.can_execute, plan.blockers
    assert Path(plan.output_dir).resolve() != Path(job.output_path).parent.resolve()


def test_target_name_validated(env):
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="--help"), env.scan)
    assert not plan.can_execute and plan.confirm_token is None
    assert any("invalid target" in b for b in plan.blockers)
    assert not any(a.upper() == "--HELP" for a in plan.argv)


def test_confirmation_required(env):
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0"), env.scan)
    with pytest.raises(PermissionError):
        env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=False, confirm_token=plan.confirm_token))
    with pytest.raises(PermissionError):
        env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=True, confirm_token="wrong"))
    job = env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=True, confirm_token=plan.confirm_token))
    wait(env.mgr, job.id)
    with pytest.raises(LookupError):  # single use
        env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=True, confirm_token=plan.confirm_token))


def test_source_changed_after_plan(env):
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0"), env.scan)
    with open(env.src, "ab") as fh:
        fh.write(b"\0" * 32)
    with pytest.raises(RuntimeError, match="changed"):
        env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=True, confirm_token=plan.confirm_token))


def test_cancel(env, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "1")
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0"), env.scan)
    job = env.mgr.execute(ExecuteRequest(plan_id=plan.plan_id, confirm=True, confirm_token=plan.confirm_token))
    time.sleep(0.5)
    # a second heavy job is refused while one is running
    plan2 = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q8_0"), env.scan)
    with pytest.raises(RuntimeError, match="already running"):
        env.mgr.execute(ExecuteRequest(plan_id=plan2.plan_id, confirm=True, confirm_token=plan2.confirm_token))
    env.mgr.cancel(job.id)
    j = wait(env.mgr, job.id)
    assert j.status == "cancelled"
    assert not Path(plan.partial_path).exists() and not Path(plan.output_path).exists()


@pytest.mark.parametrize(
    "kw,blocker",
    [
        ({"target": "IQ2_XS"}, "importance matrix"),
        ({"target": "Q5_1"}, "not offered"),
        ({"target": "F16"}, "already F16"),
        ({"target": "Q4_0", "output_dir": "relative/dir"}, "absolute"),
    ],
)
def test_quantize_blockers(env, kw, blocker):
    plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", **kw), env.scan)
    assert not plan.can_execute and plan.confirm_token is None
    assert any(blocker in b for b in plan.blockers), plan.blockers


def test_output_dir_must_be_separate(env):
    for out in (env.src.parent, env.src.parent / "sub"):
        plan = env.mgr.create_plan(PlanRequest(scan_id=env.scan.id, operation="quantize", target="Q4_0", output_dir=str(out)), env.scan)
        assert any("separate from the source" in b for b in plan.blockers)


def test_requantize_requires_opt_in(env):
    q = llama_like_gguf(env.tmp / "models" / "q.gguf", wtype="Q8_0", ftype=7)
    s = scan_path(str(q), env.settings)
    plan = env.mgr.create_plan(PlanRequest(scan_id=s.id, operation="quantize", target="Q4_0"), s)
    assert any("allow requantize" in b for b in plan.blockers)
    plan = env.mgr.create_plan(PlanRequest(scan_id=s.id, operation="quantize", target="Q4_0", allow_requantize=True), s)
    assert plan.can_execute and "--allow-requantize" in plan.argv
    assert any("requantizing" in w for w in plan.warnings)


def test_convert_blockers(env):
    d = hf_llama_dir(env.tmp / "hf", shards=1)
    s = scan_path(str(d), env.settings)
    plan = env.mgr.create_plan(PlanRequest(scan_id=s.id, operation="convert", target="f16"), s)
    assert any("converter unavailable" in b for b in plan.blockers)
    # quantize is not offered for HF input
    plan = env.mgr.create_plan(PlanRequest(scan_id=s.id, operation="quantize", target="Q4_0"), s)
    assert any("GGUF" in b for b in plan.blockers)


def test_convert_blocks_custom_code(env, tmp_path):
    env.report.convert = ConvertTool(available=True, script=str(tmp_path / "convert_hf_to_gguf.py"), python=sys.executable, outtypes=["f16", "bf16", "q8_0", "auto"])
    d = hf_llama_dir(env.tmp / "hf", shards=1)
    (d / "modeling_evil.py").write_text("import os")
    s = scan_path(str(d), env.settings)
    plan = env.mgr.create_plan(PlanRequest(scan_id=s.id, operation="convert", target="f16"), s)
    assert any("custom Python code" in b for b in plan.blockers)
    (d / "modeling_evil.py").unlink()
    s = scan_path(str(d), env.settings)
    plan = env.mgr.create_plan(PlanRequest(scan_id=s.id, operation="convert", target="f16"), s)
    assert plan.can_execute, plan.blockers
    assert plan.argv[:2] == [sys.executable, str(tmp_path / "convert_hf_to_gguf.py")]
    assert plan.argv[plan.argv.index("--outtype") + 1] == "f16"
    assert plan.estimated_output_bytes


def test_interrupted_jobs_marked_on_restart(env):
    env.history.save_job("abc", "2026-01-01T00:00:00", "running", {"id": "abc", "status": "running"})
    h2 = History(env.settings.db_path)
    assert h2.get_job("abc")["status"] == "interrupted"
