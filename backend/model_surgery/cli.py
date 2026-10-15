"""Command-line entry point: ``model-surgery serve | inspect | tools``."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import webbrowser

from .config import load_settings


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="model-surgery", description="Local LLM/SLM inspection and optimization workbench")
    ap.add_argument("--data-dir", help="history, config and default output location (default ~/.model_surgery)")
    ap.add_argument("--llama-cpp-dir", help="directory containing llama-quantize")
    ap.add_argument("--convert-script", help="path to llama.cpp convert_hf_to_gguf.py")
    ap.add_argument("--convert-python", help="Python interpreter with the convert requirements")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--allow-remote", action="store_true", help="permit binding to a non-loopback address (exposes your filesystem listing!)")
    s.add_argument("--open", action="store_true", help="open the browser")

    i = sub.add_parser("inspect", help="scan a model and print a report")
    i.add_argument("path")
    i.add_argument("--format", choices=["summary", "json", "markdown"], default="summary")

    sub.add_parser("tools", help="show detected llama.cpp tooling")

    args = ap.parse_args(argv)
    settings = load_settings(
        data_dir=args.data_dir,
        llama_cpp_bin_dir=args.llama_cpp_dir,
        convert_script=args.convert_script,
        convert_python=args.convert_python,
    )

    if args.cmd == "serve":
        if not _is_loopback(args.host):
            if not args.allow_remote:
                print("refusing to bind to a non-loopback address without --allow-remote", file=sys.stderr)
                return 2
            print("WARNING: Model Surgery has no authentication; anyone who can reach this port can browse your files.", file=sys.stderr)
            settings.extra_allowed_hosts.append(args.host)
        import uvicorn

        from .api import create_app

        settings.host, settings.port = args.host, args.port
        app = create_app(settings)
        url = f"http://{'localhost' if _is_loopback(args.host) else args.host}:{args.port}/"
        print(f"Model Surgery on {url}  (data: {settings.data_dir})")
        if args.open:
            webbrowser.open(url)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    if args.cmd == "inspect":
        from .reports import fmt_bytes, fmt_count, markdown_report
        from .scanner import ScanError, scan_path

        try:
            r = scan_path(args.path, settings)
        except ScanError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.format == "json":
            print(r.model_dump_json(indent=2))
        elif args.format == "markdown":
            print(markdown_report(r))
        else:
            fmt = f"{r.format.name} {r.format.version}" if r.format.version else r.format.name
            print(f"{r.name}  [{fmt}]  {fmt_bytes(r.total_size_bytes)}")
            print(f"integrity: {'ok' if r.integrity.valid else 'FAILED'}  quantization: {r.quantization.label}")
            for f in r.hyperparameters:
                v = fmt_count(f.value) if f.key.startswith("n_params") else f.value
                print(f"  {f.label:<30} {v!s:<28} [{f.provenance}] {f.source}")
            for e in r.integrity.errors[:20]:
                print(f"  ERROR: {e}")
        return 0 if r.integrity.valid else 3

    if args.cmd == "tools":
        from .tooling import detect_tools

        print(json.dumps(detect_tools(settings, refresh=True).model_dump(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
