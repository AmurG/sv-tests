#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Hybrid AmurG sv-tests dashboard builder.
#
# This starts from an upstream sv-tests-results artifact, recomputes the
# CIRCT-dependent columns against a selected CIRCT remote/ref, and splices those
# rows plus rendered logs back into a static sv-tests-results checkout.

import argparse
import concurrent.futures
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


CSV_FIELDS = [
    "TestName",
    "Tool",
    "Group",
    "Pass",
    "ExitCode",
    "Tags",
    "InputBytes",
    "AllowedTimeout",
    "TimeUser",
    "TimeSystem",
    "TimeWall",
    "RamUsageMiB",
]


@dataclass(frozen=True)
class ToolReplacement:
    csv_tool: str
    log_dir: str
    result_dir: str
    display_name: str
    cell_slug: str
    report_dir: Path


def run(cmd, cwd=None, env=None, check=True):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def capture(cmd, cwd=None, env=None):
    return subprocess.check_output(cmd, cwd=cwd, env=env, text=True).strip()


def read_csv(path):
    with path.open(newline="", errors="ignore") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def github_tree_url(remote, ref):
    parsed = urlparse(remote)
    if parsed.netloc == "github.com":
        path = parsed.path[:-4] if parsed.path.endswith(".git") else parsed.path
        return f"https://github.com{path}/tree/{ref}"
    return remote


def baseline_circt_ref(repo):
    line = capture(["git", "ls-tree", "HEAD", "third_party/tools/circt-verilog"],
                   cwd=repo)
    parts = line.split()
    if len(parts) < 3:
        raise RuntimeError("could not read CIRCT submodule ref from HEAD")
    return parts[2]


def prepare_results_checkout(work_dir, baseline_url, baseline_ref):
    results_dir = work_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    run([
        "git",
        "clone",
        "--single-branch",
        "--depth",
        "1",
        "--branch",
        baseline_ref,
        baseline_url,
        str(results_dir),
    ])
    baseline_sha = capture(["git", "rev-parse", "HEAD"], cwd=results_dir)
    return results_dir, baseline_sha


def prepare_circt(repo, circt_remote, circt_ref):
    circt_dir = repo / "third_party" / "tools" / "circt-verilog"
    if not (circt_dir / ".git").exists():
        run(["git", "submodule", "update", "--init", "third_party/tools/circt-verilog"],
            cwd=repo)

    run(["git", "remote", "remove", "amurg-target"], cwd=circt_dir, check=False)
    run(["git", "remote", "add", "amurg-target", circt_remote], cwd=circt_dir)
    run(["git", "fetch", "--depth", "1", "amurg-target", circt_ref],
        cwd=circt_dir)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=circt_dir)
    run(["git", "submodule", "sync", "--recursive"], cwd=circt_dir)
    run([
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--depth",
        "1",
        "llvm",
    ],
        cwd=circt_dir)

    resolved = capture(["git", "rev-parse", "HEAD"], cwd=circt_dir)
    llvm_ref = capture(["git", "rev-parse", "HEAD"], cwd=circt_dir / "llvm")
    return circt_dir, resolved, llvm_ref


def build_circt(circt_dir, build_dir, build_jobs):
    build_dir.mkdir(parents=True, exist_ok=True)
    run([
        "cmake",
        str(circt_dir / "llvm" / "llvm"),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_EXTERNAL_PROJECTS=circt",
        f"-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR={circt_dir}",
        "-DCIRCT_SLANG_FRONTEND_ENABLED=ON",
        "-DLLVM_USE_LINKER=lld",
        "-DVERILATOR_DISABLE=ON",
    ],
        cwd=circt_dir)
    run([
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "circt-verilog",
        "arcilator",
        f"-j{build_jobs}",
    ],
        cwd=circt_dir)


def dashboard_env(repo, out_dir, circt_build_dir, circt_remote, circt_ref):
    env = os.environ.copy()
    env.update({
        "OUT_DIR": str(out_dir),
        "CONF_DIR": str(repo / "conf"),
        "TESTS_DIR": str(repo / "tests"),
        "RUNNERS_DIR": str(repo / "tools" / "runners"),
        "THIRD_PARTY_DIR": str(repo / "third_party"),
        "CIRCT_BIN_DIR": str(circt_build_dir / "bin"),
        "CIRCT_REMOTE": circt_remote,
        "CIRCT_REF": circt_ref,
    })
    env["PATH"] = str(circt_build_dir / "bin") + os.pathsep + env.get("PATH", "")
    return env


def init_git_transport():
    run([
        "git",
        "config",
        "--global",
        "url.https://github.com/.insteadOf",
        "git://github.com/",
    ])


def init_uvm_submodules(repo):
    init_git_transport()
    run([
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--depth",
        "1",
        "--force",
        "third_party/tests/easyUVM",
        "third_party/tests/uvm",
    ],
        cwd=repo)


def init_full_sv_tests_submodules(repo):
    init_git_transport()
    run([
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--depth",
        "1",
        "--force",
        "third_party/tests",
        "third_party/cores",
        "third_party/tools/yosys",
        "third_party/tools/icarus",
    ],
        cwd=repo)


def generate_uvm_tests(repo, env, jobs):
    run(["make", "generate-template_generator", f"-j{jobs}"], cwd=repo, env=env)
    run(["make", "generate-easyUVM", f"-j{jobs}"], cwd=repo, env=env)


def test_name_to_paths(tests_dir):
    name_to_paths = {}
    for path in sorted(tests_dir.rglob("*.sv")):
        text = path.read_text(errors="ignore")
        match = re.search(r"^:name:\s*(.+)$", text, re.MULTILINE)
        name = match.group(1).strip() if match else str(path.relative_to(tests_dir))
        name_to_paths.setdefault(name, []).append(str(path.relative_to(tests_dir)))
    return name_to_paths


def write_exact_uvm_list(results_csv, tests_dir, out_txt, out_json):
    name_to_paths = test_name_to_paths(tests_dir)
    rows = [
        row for row in read_csv(results_csv)
        if row["Tool"] == "Verilator" and "uvm" in row["Tags"].split()
    ]

    selected = []
    missing = []
    ambiguous = {}
    for row in rows:
        paths = name_to_paths.get(row["TestName"], [])
        if not paths:
            missing.append(row["TestName"])
        elif len(paths) > 1:
            ambiguous[row["TestName"]] = paths
        else:
            selected.append(paths[0])

    summary = {
        "source_report": str(results_csv),
        "source_tool": "Verilator",
        "selection": "rows with exact uvm token in Tags",
        "rows": len(rows),
        "selected": len(selected),
        "missing": missing,
        "ambiguous": ambiguous,
    }
    out_txt.write_text("\n".join(selected) + "\n")
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if missing or ambiguous:
        raise RuntimeError(json.dumps(summary, indent=2, sort_keys=True))
    return selected


def run_runner_test(repo, env, out_dir, runner, test):
    log_path = out_dir / "logs" / runner / f"{test}.log"
    cmd = [
        "./tools/runner",
        "--runner",
        runner,
        "--test",
        test,
        "--out",
        str(log_path),
        "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=repo, env=env)
    return test, proc.returncode


def run_runner_list_report(repo, env, out_dir, runner, tests, jobs):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "logs" / runner).mkdir(parents=True, exist_ok=True)

    for flag, filename in [("--version", "version"), ("--url", "url")]:
        run([
            "./tools/runner",
            "--runner",
            runner,
            flag,
            "--out",
            str(out_dir / "logs" / runner / filename),
        ],
            cwd=repo,
            env=env)

    failures = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(run_runner_test, repo, env, out_dir, runner, test)
            for test in tests
        ]
        for future in concurrent.futures.as_completed(futures):
            test, rc = future.result()
            completed += 1
            if rc != 0:
                failures.append((test, rc))
            if completed % 25 == 0 or completed == len(tests):
                print(f"{runner}: completed {completed}/{len(tests)}",
                      flush=True)
    if failures:
        raise RuntimeError(f"{runner} infrastructure failures: {failures[:10]}")

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    revision = capture(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    run([
        "./tools/sv-report",
        "--logs",
        str(out_dir / "logs"),
        "--out",
        str(report_dir / "index.html"),
        "--csv",
        str(report_dir / "report.csv"),
        "--revision",
        revision,
    ],
        cwd=repo,
        env=env)
    for pattern in ["*.css", "*.js", "*.png", "*.svg"]:
        for path in (repo / "conf" / "report").glob(pattern):
            shutil.copy2(path, report_dir / path.name)
    return report_dir


def run_official_generator_loop(repo, env, out_dir, runner_filter, jobs):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    env = dict(env)
    env["RUNNERS_FILTER"] = runner_filter
    big = set(env.get("BIG_GENERATORS", "fusesoc black-parrot").split())
    generators = capture(["make", "list-generators"], cwd=repo, env=env).split()

    run(["make", "info"], cwd=repo, env=env)
    for gen in [g for g in generators if g not in big]:
        run(["make", f"generate-{gen}", f"-j{jobs}"], cwd=repo, env=env)
        run(["make", "tests", "versions", "urls", f"-j{jobs}"],
            cwd=repo,
            env=env)
    for gen in [g for g in generators if g in big]:
        run(["make", f"generate-{gen}"], cwd=repo, env=env)
        run(["make", "tests", "versions", "urls"], cwd=repo, env=env)

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    revision = capture(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    run([
        "./tools/sv-report",
        "--logs",
        str(out_dir / "logs"),
        "--out",
        str(report_dir / "index.html"),
        "--csv",
        str(report_dir / "report.csv"),
        "--revision",
        revision,
    ],
        cwd=repo,
        env=env)
    for pattern in ["*.css", "*.js", "*.png", "*.svg"]:
        for path in (repo / "conf" / "report").glob(pattern):
            shutil.copy2(path, report_dir / path.name)
    return report_dir


def summarize_by_tag(rows):
    summary = defaultdict(lambda: {"passed": 0, "total": 0})
    for row in rows:
        for tag in row["Tags"].split():
            summary[tag]["total"] += 1
            if row["Pass"] == "True":
                summary[tag]["passed"] += 1
    return dict(summary)


def make_cell(tag, summary, cell_slug):
    if tag not in summary:
        return "                  <td></td>"

    passed = summary[tag]["passed"]
    total = summary[tag]["total"]
    text = f"{passed}/{total}"
    if total == 0:
        return "                  <td></td>"
    if passed == total:
        return (
            f'                  <td class="test-passed" '
            f'data-tool="{cell_slug}">{text}</td>')
    if passed == 0:
        return (
            f'                  <td class="test-failed" '
            f'data-tool="{cell_slug}">{text}</td>')
    percent = round(passed * 100 / total)
    return (
        f'                  <td class="test-varied" '
        f'data-tool="{cell_slug}" style="--val: {percent}%">{text}</td>')


def load_tool_names(index):
    match = re.search(r"var TOOL_NAMES = (\[.*?\]);", index)
    if not match:
        raise RuntimeError("TOOL_NAMES not found")
    return match, json.loads(match.group(1))


def set_tool_names(index, tools):
    match, _ = load_tool_names(index)
    replacement = "var TOOL_NAMES = " + json.dumps(tools) + ";"
    index = index[:match.start()] + replacement + index[match.end():]
    index = re.sub(r"--TOOLS_COUNT:\s*\d+", f"--TOOLS_COUNT: {len(tools)}",
                   index)
    return index


def ensure_tool_name(index, display_name, after="Verilator"):
    match, tools = load_tool_names(index)
    if display_name not in tools:
        insert_at = tools.index(after) + 1 if after in tools else len(tools)
        tools.insert(insert_at, display_name)
    return set_tool_names(index, tools)


def ensure_filter(index, display_name, cell_slug, after_display="Verilator"):
    if f'value="{cell_slug}"' in index:
        return index
    label = (
        f'                        <label title="{html.escape(display_name)}">'
        f'<input type="checkbox" value="{cell_slug}" checked>'
        f'{html.escape(display_name)}</label>')
    after_re = re.compile(
        r'(\n\s*<label title="' + re.escape(after_display) +
        r'"><input type="checkbox" value="' + re.escape(after_display.lower()) +
        r'" checked>' + re.escape(after_display) + r'</label>)')
    index, count = after_re.subn(lambda m: m.group(1) + "\n" + label, index, 1)
    if count == 0:
        raise RuntimeError(f"filter insertion point {after_display} not found")
    return index


def normalize_tool_header_z(index):
    header_pattern = re.compile(r"(<thead>[\s\S]*?</thead>)")
    th_pattern = re.compile(
        r'(<th title="[^"]*" style="--z: )\d+(">[\s\S]*?</th>)')

    def replace_header(match):
        header = match.group(1)
        matches = list(th_pattern.finditer(header))
        count = len(matches)
        if count == 0:
            return header
        pieces = []
        pos = 0
        for idx, th_match in enumerate(matches):
            pieces.append(header[pos:th_match.start()])
            pieces.append(th_match.group(1))
            pieces.append(str(count - idx - 1))
            pieces.append(th_match.group(2))
            pos = th_match.end()
        pieces.append(header[pos:])
        return "".join(pieces)

    return header_pattern.sub(replace_header, index)


def tool_header(display_name, version, url):
    title = html.escape(version, quote=True).replace("\n", "&#10;")
    href = html.escape(url, quote=True)
    return (
        f'\n                  <th title="{title}" style="--z: 0">\n'
        f'                     <a class="tool_link" target="_blank" '
        f'href="{href}">{html.escape(display_name)}</a>\n'
        f'                  </th>')


def ensure_or_update_header(index, display_name, version, url, after="Verilator"):
    title = html.escape(version, quote=True).replace("\n", "&#10;")
    href = html.escape(url, quote=True)
    display_re = re.escape(display_name)
    existing = re.compile(
        r'(<th title=")[^"]*(" style="--z: )\d+(">[\s\S]*?'
        r'<a class="tool_link" target="_blank" href=")[^"]*(">'
        + display_re + r'</a>\n\s*</th>)')
    index, count = existing.subn(
        lambda m: m.group(1) + title + m.group(2) + "0" + m.group(3) + href
        + m.group(4),
        index)
    if count > 0:
        return index

    header = tool_header(display_name, version, url)
    after_re = re.compile(
        r'(\n\s*<th title="[^"]*" style="--z: \d+">\n'
        r'\s*<a class="tool_link" target="_blank" href="[^"]+">'
        + re.escape(after) + r'</a>\n\s*</th>)')
    index, count = after_re.subn(lambda m: m.group(1) + header, index)
    if count == 0:
        raise RuntimeError(f"header insertion point {after} not found")
    return index


def replace_or_insert_cells(index, summary, cell_slug, after_slug="verilator"):
    row_pattern = re.compile(
        r'(<tr\b(?=[^>]*\bdata-tag="([^"]+)")[\s\S]*?</tr>)')
    existing_pattern = re.compile(
        r'\n\s*<td[^>]*\bdata-tool="' + re.escape(cell_slug) +
        r'"[^>]*>[\s\S]*?</td>')
    after_pattern = re.compile(
        r'\n\s*<td[^>]*\bdata-tool="' + re.escape(after_slug) +
        r'"[^>]*>[\s\S]*?</td>')

    def replace_row(match):
        row = match.group(1)
        tag = match.group(2)
        new_cell = "\n" + make_cell(tag, summary, cell_slug)
        existing_match = existing_pattern.search(row)
        if existing_match:
            return row[:existing_match.start()] + new_cell + row[existing_match.end():]
        after_match = after_pattern.search(row)
        if after_match:
            return row[:after_match.end()] + new_cell + row[after_match.end():]
        return row

    index, count = row_pattern.subn(replace_row, index)
    if count == 0:
        raise RuntimeError("data-tag rows not found")
    return index


def copy_tree_contents(src, dst):
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def read_tool_text(out_dir, log_dir, name):
    path = out_dir / "logs" / log_dir / name
    if path.exists():
        return path.read_text(errors="ignore").strip()
    return ""


def integrate_reports(results_dir, replacements, metadata):
    all_replacement_tools = {tool.csv_tool for tool in replacements}
    rows = [
        row for row in read_csv(results_dir / "report.csv")
        if row["Tool"] not in all_replacement_tools
    ]

    index_path = results_dir / "index.html"
    index = index_path.read_text()

    for tool in replacements:
        generated_csv = tool.report_dir / "report.csv"
        tool_rows = [
            row for row in read_csv(generated_csv)
            if row["Tool"] == tool.csv_tool
        ]
        if not tool_rows:
            raise RuntimeError(f"{generated_csv} has no {tool.csv_tool} rows")
        rows.extend(tool_rows)

        for dst in [
                results_dir / "logs" / tool.log_dir,
                results_dir / "results" / tool.result_dir,
        ]:
            if dst.exists():
                shutil.rmtree(dst)
        copy_tree_contents(tool.report_dir / "logs" / tool.log_dir,
                           results_dir / "logs" / tool.log_dir)
        copy_tree_contents(tool.report_dir / "results" / tool.result_dir,
                           results_dir / "results" / tool.result_dir)
        copy_tree_contents(tool.report_dir / "tests", results_dir / "tests")
        copy_tree_contents(tool.report_dir / "third_party",
                           results_dir / "third_party")

        version = read_tool_text(tool.report_dir.parent, tool.log_dir, "version")
        url = read_tool_text(tool.report_dir.parent, tool.log_dir, "url")
        if not version:
            version = tool.display_name
        if not url:
            url = metadata.get("circt_url", "")

        if tool.display_name == "Arcilator":
            index = ensure_tool_name(index, tool.display_name)
            index = ensure_filter(index, tool.display_name, tool.cell_slug)
        index = ensure_or_update_header(index, tool.display_name, version, url)
        index = replace_or_insert_cells(
            index, summarize_by_tag(tool_rows), tool.cell_slug)

    write_csv(results_dir / "report.csv", rows)
    index = normalize_tool_header_z(index)
    index_path.write_text(index)
    (results_dir / "amurg-hybrid-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def commit_and_push_results(results_dir, push_url, branch, message):
    if push_url:
        print("+ git remote set-url origin <masked>", flush=True)
        subprocess.run(["git", "remote", "set-url", "origin", push_url],
                       cwd=results_dir,
                       check=True)
    run(["git", "config", "user.name", "amurg-hybrid-dashboard"],
        cwd=results_dir)
    run(["git", "config", "user.email", "actions@github.com"], cwd=results_dir)
    run(["git", "add", "-A"], cwd=results_dir)
    status = capture(["git", "status", "--short"], cwd=results_dir)
    if not status:
        print("No result changes to commit.", flush=True)
        return False

    run(["git", "fetch", "origin", branch], cwd=results_dir, check=False)
    remote_parent = capture(["git", "rev-parse", "--verify", f"origin/{branch}"],
                            cwd=results_dir)
    tree = capture(["git", "write-tree"], cwd=results_dir)
    commit = capture(["git", "commit-tree", tree, "-p", remote_parent, "-m", message],
                     cwd=results_dir)
    run(["git", "push", "origin", f"{commit}:{branch}"], cwd=results_dir)
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results-url",
                        default=os.environ.get(
                            "BASELINE_RESULTS_URL",
                            "https://github.com/chipsalliance/sv-tests-results.git"))
    parser.add_argument("--baseline-results-ref",
                        default=os.environ.get("BASELINE_RESULTS_REF",
                                               "gh-pages"))
    parser.add_argument("--results-push-url",
                        default=os.environ.get("RESULTS_PUSH_URL", ""))
    parser.add_argument("--results-push-branch",
                        default=os.environ.get("RESULTS_PUSH_BRANCH",
                                               "gh-pages"))
    parser.add_argument("--circt-remote",
                        default=os.environ.get("CIRCT_REMOTE",
                                               "https://github.com/llvm/circt.git"))
    parser.add_argument("--circt-ref", default=os.environ.get("CIRCT_REF", ""))
    parser.add_argument("--work-dir",
                        default=os.environ.get("AMURG_WORK_DIR",
                                               "out/amurg-hybrid-dashboard"))
    parser.add_argument("--circt-build-dir",
                        default=os.environ.get("CIRCT_BUILD_DIR", ""))
    parser.add_argument("--jobs",
                        type=int,
                        default=int(os.environ.get("JOBS", "16")))
    parser.add_argument("--build-jobs",
                        type=int,
                        default=int(os.environ.get("BUILD_JOBS", "32")))
    parser.add_argument("--arcilator-scope",
                        choices=["uvm", "all", "none"],
                        default=os.environ.get("ARCILATOR_SCOPE", "all"))
    parser.add_argument("--skip-circt-verilog",
                        action="store_true",
                        default=os.environ.get("SKIP_CIRCT_VERILOG", "") == "1")
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    work_dir = (repo / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    circt_ref = args.circt_ref or baseline_circt_ref(repo)
    results_dir, baseline_results_sha = prepare_results_checkout(
        work_dir, args.baseline_results_url, args.baseline_results_ref)

    circt_dir, resolved_circt_ref, llvm_ref = prepare_circt(
        repo, args.circt_remote, circt_ref)
    build_dir = Path(args.circt_build_dir
                     or work_dir / "circt-build").resolve()
    build_circt(circt_dir, build_dir, args.build_jobs)

    needs_full_sv_tests = (
        (not args.skip_circt_verilog) or args.arcilator_scope == "all")
    if needs_full_sv_tests:
        init_full_sv_tests_submodules(repo)
    elif args.arcilator_scope == "uvm":
        init_uvm_submodules(repo)

    replacements = []
    common = dashboard_env(repo, work_dir / "unused", build_dir,
                           args.circt_remote, resolved_circt_ref)

    if not args.skip_circt_verilog:
        out_dir = work_dir / "out_circt_verilog"
        env = dashboard_env(repo, out_dir, build_dir, args.circt_remote,
                            resolved_circt_ref)
        report_dir = run_official_generator_loop(repo, env, out_dir,
                                                 "circt_verilog", args.jobs)
        replacements.append(
            ToolReplacement("circt_verilog", "circt_verilog", "circt_verilog",
                            "circt_verilog", "circt_verilog", report_dir))

    if args.arcilator_scope != "none":
        out_dir = work_dir / f"out_arcilator_{args.arcilator_scope}"
        env = dashboard_env(repo, out_dir, build_dir, args.circt_remote,
                            resolved_circt_ref)
        if args.arcilator_scope == "uvm":
            generate_uvm_tests(repo, common, args.jobs)
            list_txt = work_dir / "exact_uvm_tests.txt"
            list_json = work_dir / "exact_uvm_tests.summary.json"
            tests = write_exact_uvm_list(results_dir / "report.csv",
                                         repo / "tests", list_txt, list_json)
            report_dir = run_runner_list_report(repo, env, out_dir, "Arcilator",
                                                tests, args.jobs)
        else:
            report_dir = run_official_generator_loop(repo, env, out_dir,
                                                     "Arcilator", args.jobs)
        replacements.append(
            ToolReplacement("Arcilator", "Arcilator", "arcilator",
                            "Arcilator", "arcilator", report_dir))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_results_url": args.baseline_results_url,
        "baseline_results_ref": args.baseline_results_ref,
        "baseline_results_sha": baseline_results_sha,
        "sv_tests_repo": capture(["git", "config", "--get", "remote.origin.url"],
                                 cwd=repo),
        "sv_tests_sha": capture(["git", "rev-parse", "HEAD"], cwd=repo),
        "circt_remote": args.circt_remote,
        "circt_requested_ref": circt_ref,
        "circt_resolved_ref": resolved_circt_ref,
        "circt_llvm_ref": llvm_ref,
        "circt_url": github_tree_url(args.circt_remote, resolved_circt_ref),
        "replaced_tools": [tool.csv_tool for tool in replacements],
        "arcilator_scope": args.arcilator_scope,
        "quality_note": (
            "Non-CIRCT columns are imported from the baseline results artifact. "
            "circt_verilog and Arcilator rows are recomputed through sv-tests "
            "tools/runner. Arcilator uses the same expected-failure and "
            "simulation parseLog machinery as Verilator; simulation tests run "
            "arcilator --run and fail conservatively if the current CIRCT/Arc "
            "pipeline cannot execute them."),
    }
    integrate_reports(results_dir, replacements, metadata)

    if args.push:
        message = (
            "Deploy AmurG hybrid CIRCT dashboard\n\n"
            f"Baseline results: {baseline_results_sha}\n"
            f"CIRCT: {args.circt_remote}@{resolved_circt_ref}\n"
            f"LLVM: {llvm_ref}\n"
            f"Replaced tools: {', '.join(metadata['replaced_tools'])}\n"
            f"Arcilator scope: {args.arcilator_scope}")
        commit_and_push_results(results_dir, args.results_push_url,
                                args.results_push_branch, message)
    else:
        print(f"Hybrid results staged at {results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
