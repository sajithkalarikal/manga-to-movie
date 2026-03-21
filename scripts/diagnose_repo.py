#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import tokenize
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "diagnostics"
DEFAULT_JSON = OUTPUT_DIR / "repo_diagnostics.json"
DEFAULT_MD = OUTPUT_DIR / "repo_diagnostics.md"

IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "outputs",
    "models",
    "logs",
    ".model_cache",
}
SOURCE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".html",
    ".sh",
    ".md",
    ".json",
}
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}
WEB_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".css"}
LEGACY_FRONTEND_EXTENSIONS = {".html", ".js", ".css"}
RESOLVABLE_WEB_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".css")
HTML_REFERENCE_RE = re.compile(r"""(?:src|href)=["']([^"']+)["']""")
JS_IMPORT_RE = re.compile(
    r"""(?:import|export)\s+(?:[^'"]*?\sfrom\s+)?["']([^"']+)["']|import\(\s*["']([^"']+)["']\s*\)""",
    re.MULTILINE,
)
CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\()?['"]([^'"]+)['"]\)?""")
JS_DEBUG_RE = re.compile(r"""\bconsole\.(?:log|debug|info|warn|error)\s*\(|\bdebugger\b""")
PY_DEBUG_RE = re.compile(r"""\bprint\s*\(|\bbreakpoint\s*\(|\bpdb\.set_trace\s*\(""")
JS_CONSTANT_BRANCH_RE = re.compile(
    r"""\bif\s*\(\s*(?:false|0|null|undefined)\s*\)|\bif\s*\(\s*true\s*\)|\bwhile\s*\(\s*false\s*\)"""
)
TEST_FILE_PATTERNS = (
    "test_*.py",
    "*_test.py",
    "*.test.ts",
    "*.test.tsx",
    "*.spec.ts",
    "*.spec.tsx",
)


@dataclass
class ReachabilityResult:
    entrypoints: list[Path]
    reachable: set[Path]
    orphan_candidates: list[Path]
    dependency_edges: dict[Path, set[Path]]


@dataclass
class BranchFinding:
    path: Path
    line_start: int
    line_end: int
    reason: str

    @property
    def line_count(self) -> int:
        return max(1, self.line_end - self.line_start + 1)


@dataclass
class LineMetrics:
    total_lines: int
    blank_lines: int
    comment_lines: int
    executable_lines: int


def is_ignored(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in IGNORE_DIRS for part in relative.parts)


def walk_repo_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if is_ignored(path, root):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def extension_inventory(files: Iterable[Path]) -> dict[str, int]:
    counts = Counter(path.suffix or "<no_ext>" for path in files)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def directory_inventory(files: Iterable[Path], root: Path) -> dict[str, int]:
    counts = Counter(path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "." for path in files)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def python_module_keys(path: Path, root: Path) -> set[str]:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    keys: set[str] = set()
    if parts[-1] == "__init__":
        keys.add(".".join(parts[:-1]))
    else:
        keys.add(".".join(parts))
    if len(parts) == 1:
        keys.add(parts[0])
    if parts[0] == "scripts" and len(parts) == 2:
        keys.add(parts[1])
    return {key for key in keys if key}


def build_python_module_index(py_files: Iterable[Path], root: Path) -> dict[str, Path]:
    module_index: dict[str, Path] = {}
    for path in py_files:
        for key in python_module_keys(path, root):
            module_index.setdefault(key, path)
    return module_index


def resolve_python_import(
    *,
    current_file: Path,
    module_name: str | None,
    level: int,
    module_index: dict[str, Path],
    root: Path,
) -> str | None:
    candidates: list[str] = []
    current_module = next(iter(sorted(python_module_keys(current_file, root))))
    current_parts = current_module.split(".")
    package_parts = current_parts[:-1]

    if level > 0:
        anchor = package_parts[: len(package_parts) - (level - 1)]
        if module_name:
            candidates.append(".".join([*anchor, module_name]))
        else:
            candidates.append(".".join(anchor))
    elif module_name:
        candidates.append(module_name)

    if module_name:
        if module_name in module_index:
            candidates.append(module_name)
        if "." in module_name:
            parts = module_name.split(".")
            for size in range(len(parts) - 1, 0, -1):
                candidates.append(".".join(parts[:size]))
        if current_file.parent.name == "scripts":
            candidates.append(f"scripts.{module_name}")
        candidates.append(module_name.split(".")[0])

    for candidate in candidates:
        if candidate in module_index:
            return candidate
    return None


def python_dependencies(path: Path, module_index: dict[str, Path], root: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return set()

    dependencies: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = resolve_python_import(
                    current_file=path,
                    module_name=alias.name,
                    level=0,
                    module_index=module_index,
                    root=root,
                )
                if resolved and resolved in module_index:
                    dependencies.add(module_index[resolved])
        elif isinstance(node, ast.ImportFrom):
            base_name = node.module or ""
            resolved_base = resolve_python_import(
                current_file=path,
                module_name=base_name or None,
                level=node.level,
                module_index=module_index,
                root=root,
            )
            if resolved_base and resolved_base in module_index:
                dependencies.add(module_index[resolved_base])
            for alias in node.names:
                if alias.name == "*":
                    continue
                submodule_name = f"{base_name}.{alias.name}" if base_name else alias.name
                resolved_submodule = resolve_python_import(
                    current_file=path,
                    module_name=submodule_name or None,
                    level=node.level,
                    module_index=module_index,
                    root=root,
                )
                if resolved_submodule and resolved_submodule in module_index:
                    dependencies.add(module_index[resolved_submodule])
    return dependencies


def resolve_web_specifier(base_path: Path, specifier: str) -> Path | None:
    if not specifier.startswith("."):
        return None
    candidate = (base_path.parent / specifier).resolve()
    direct_candidates = [candidate]
    if candidate.suffix:
        direct_candidates.extend(candidate / f"index{suffix}" for suffix in RESOLVABLE_WEB_EXTENSIONS)
    else:
        direct_candidates.extend(candidate.with_suffix(suffix) for suffix in RESOLVABLE_WEB_EXTENSIONS)
        direct_candidates.extend(candidate / f"index{suffix}" for suffix in RESOLVABLE_WEB_EXTENSIONS)
    for item in direct_candidates:
        if item.exists() and item.is_file():
            return item
    return None


def web_dependencies(path: Path) -> set[Path]:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    matches = JS_IMPORT_RE.findall(contents)
    dependencies: set[Path] = set()
    for left, right in matches:
        specifier = left or right
        if not specifier:
            continue
        resolved = resolve_web_specifier(path, specifier)
        if resolved is not None:
            dependencies.add(resolved)
    if path.suffix == ".css":
        for specifier in CSS_IMPORT_RE.findall(contents):
            resolved = resolve_web_specifier(path, specifier)
            if resolved is not None:
                dependencies.add(resolved)
    return dependencies


def html_dependencies(path: Path) -> set[Path]:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    dependencies: set[Path] = set()
    for match in HTML_REFERENCE_RE.findall(contents):
        if match.startswith(("http://", "https://", "//", "#")):
            continue
        normalized = match.split("?", 1)[0]
        if normalized.startswith("/static/"):
            target = (path.parent / Path(normalized).name).resolve()
        else:
            target = (path.parent / normalized.lstrip("/")).resolve()
        if target.exists() and target.is_file():
            dependencies.add(target)
    return dependencies


def compute_reachability(files: set[Path], entrypoints: list[Path], dependency_edges: dict[Path, set[Path]]) -> ReachabilityResult:
    reachable: set[Path] = set()
    queue = deque(path for path in entrypoints if path in files or path.exists())

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for dependency in dependency_edges.get(current, set()):
            if dependency not in reachable:
                queue.append(dependency)

    orphan_candidates = sorted(path for path in files if path not in reachable)
    return ReachabilityResult(
        entrypoints=sorted(entrypoints),
        reachable=reachable,
        orphan_candidates=orphan_candidates,
        dependency_edges=dependency_edges,
    )


def top_candidates(paths: list[Path], root: Path, limit: int = 20) -> list[str]:
    return [relpath(path, root) for path in paths[:limit]]


def discover_tests(root: Path) -> list[Path]:
    test_files: set[Path] = set()
    for pattern in TEST_FILE_PATTERNS:
        test_files.update(path for path in root.rglob(pattern) if not is_ignored(path, root))
    return sorted(test_files)


def shell_assignment(path: Path, name: str) -> str | None:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(rf'^{re.escape(name)}="([^"]+)"', contents, re.MULTILINE)
    if not match:
        return None
    return match.group(1)


def resolve_shell_path(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    normalized = value.replace("$ROOT_DIR", str(root))
    return Path(normalized).expanduser().resolve()


def analyze_models(root: Path) -> dict[str, object]:
    local_script = root / "scripts" / "local.sh"
    bubble_default = resolve_shell_path(shell_assignment(local_script, "BUBBLE_DETECTOR_DEFAULT_WEIGHTS"), root)
    panel_default = resolve_shell_path(shell_assignment(local_script, "PANEL_DETECTOR_DEFAULT_WEIGHTS"), root)

    items: list[dict[str, object]] = []
    for model_path in sorted((root / "models").rglob("*.pt")):
        status = "manual_only"
        utilized = "Only used if a script or env var points to it explicitly."
        if model_path == bubble_default:
            status = "runtime_default_bubble"
            utilized = "Auto-selected by ./load.sh and scripts/local.sh when BUBBLE_DETECTOR_WEIGHTS is unset."
        elif model_path == panel_default:
            status = "runtime_default_panel"
            utilized = "Auto-selected by ./load.sh and scripts/local.sh when PANEL_DETECTOR_WEIGHTS is unset."
        elif model_path.name.endswith(".latest.pt"):
            status = "training_resume_checkpoint"
            utilized = "Used for training resume state, not runtime inference."
        elif "archive" in model_path.parts:
            status = "archived_experiment"
            utilized = "Archived for comparison or manual recovery; not used by default runtime."

        items.append(
            {
                "path": relpath(model_path, root),
                "status": status,
                "utilized": utilized,
            }
        )

    return {
        "runtime_defaults": {
            "bubble_detector": relpath(bubble_default, root) if bubble_default and bubble_default.exists() else None,
            "panel_detector": relpath(panel_default, root) if panel_default and panel_default.exists() else None,
        },
        "files": items,
        "notes": [
            "Runtime loading goes through settings.bubble_detector_weights and settings.panel_detector_weights in config.py.",
            "The default weight selection is injected by ./load.sh and scripts/local.sh when the matching env vars are unset.",
            "Files ending in .latest.pt are training resume checkpoints, not default inference weights.",
            "Files under models/archive/ are retained experiments and are not active unless you point to them manually.",
        ],
    }


def python_line_metrics(path: Path) -> LineMetrics:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return LineMetrics(total_lines=0, blank_lines=0, comment_lines=0, executable_lines=0)

    blank_lines = sum(1 for line in contents.splitlines() if not line.strip())
    comment_lines: set[int] = set()
    code_lines: set[int] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(contents).readline):
            if token.type == tokenize.COMMENT:
                comment_lines.add(token.start[0])
            elif token.type not in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                code_lines.update(range(token.start[0], token.end[0] + 1))
    except tokenize.TokenError:
        pass

    comment_only_lines = len(comment_lines - code_lines)
    return LineMetrics(
        total_lines=len(contents.splitlines()),
        blank_lines=blank_lines,
        comment_lines=comment_only_lines,
        executable_lines=len(code_lines),
    )


def javascript_like_line_metrics(path: Path) -> LineMetrics:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return LineMetrics(total_lines=0, blank_lines=0, comment_lines=0, executable_lines=0)

    blank_lines = 0
    comment_lines = 0
    executable_lines = 0
    in_block_comment = False

    for raw_line in contents.splitlines():
        line = raw_line
        if not line.strip():
            blank_lines += 1
            continue

        has_code = False
        has_comment = False
        index = 0
        while index < len(line):
            if in_block_comment:
                has_comment = True
                end = line.find("*/", index)
                if end == -1:
                    index = len(line)
                    break
                in_block_comment = False
                index = end + 2
                continue
            if line.startswith("//", index):
                has_comment = True
                break
            if line.startswith("/*", index):
                has_comment = True
                in_block_comment = True
                index += 2
                continue
            if line[index].isspace():
                index += 1
                continue
            has_code = True
            index += 1

        if has_code:
            executable_lines += 1
        elif has_comment:
            comment_lines += 1
        else:
            blank_lines += 1

    return LineMetrics(
        total_lines=len(contents.splitlines()),
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        executable_lines=executable_lines,
    )


def line_metrics(path: Path) -> LineMetrics:
    if path.suffix == ".py":
        return python_line_metrics(path)
    if path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
        return javascript_like_line_metrics(path)
    return LineMetrics(total_lines=0, blank_lines=0, comment_lines=0, executable_lines=0)


def debug_line_hits(path: Path) -> list[int]:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    pattern = PY_DEBUG_RE if path.suffix == ".py" else JS_DEBUG_RE
    return [
        index
        for index, line in enumerate(contents.splitlines(), start=1)
        if pattern.search(line)
    ]


def _constant_bool(expr: ast.expr) -> bool | None:
    if isinstance(expr, ast.Constant):
        if expr.value in (True, False):
            return bool(expr.value)
        if expr.value in (0, None):
            return False
    return None


def _statement_lines(statement: ast.stmt) -> tuple[int, int]:
    return statement.lineno, getattr(statement, "end_lineno", statement.lineno)


def _append_dead_statements(statements: Iterable[ast.stmt], path: Path, reason: str, findings: list[BranchFinding]) -> None:
    for statement in statements:
        line_start, line_end = _statement_lines(statement)
        findings.append(
            BranchFinding(
                path=path,
                line_start=line_start,
                line_end=line_end,
                reason=reason,
            )
        )


def python_dead_branch_findings(path: Path) -> list[BranchFinding]:
    try:
        tree = ast.parse(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    findings: list[BranchFinding] = []

    def walk_body(statements: list[ast.stmt]) -> None:
        terminated = False
        for index, statement in enumerate(statements):
            if terminated:
                line_start, line_end = _statement_lines(statement)
                findings.append(
                    BranchFinding(
                        path=path,
                        line_start=line_start,
                        line_end=line_end,
                        reason="statement after return/raise/break/continue",
                    )
                )
                continue

            if isinstance(statement, ast.If):
                const_value = _constant_bool(statement.test)
                if const_value is True and statement.orelse:
                    _append_dead_statements(statement.orelse, path, "else branch under constant true condition", findings)
                    walk_body(statement.body)
                elif const_value is False and statement.body:
                    _append_dead_statements(statement.body, path, "if branch under constant false condition", findings)
                    walk_body(statement.orelse)
                else:
                    walk_body(statement.body)
                    walk_body(statement.orelse)
            elif isinstance(statement, ast.While):
                const_value = _constant_bool(statement.test)
                if const_value is False and statement.body:
                    _append_dead_statements(statement.body, path, "while loop guarded by constant false condition", findings)
                    walk_body(statement.orelse)
                else:
                    walk_body(statement.body)
                    walk_body(statement.orelse)
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.Match)):
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if isinstance(nested, list):
                        walk_body(nested)
                handlers = getattr(statement, "handlers", None)
                if isinstance(handlers, list):
                    for handler in handlers:
                        walk_body(handler.body)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk_body(statement.body)

            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                if index < len(statements) - 1:
                    terminated = True

    walk_body(tree.body)
    return findings


def javascript_constant_branch_findings(path: Path) -> list[BranchFinding]:
    try:
        contents = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[BranchFinding] = []
    for index, line in enumerate(contents.splitlines(), start=1):
        if JS_CONSTANT_BRANCH_RE.search(line):
            findings.append(
                BranchFinding(
                    path=path,
                    line_start=index,
                    line_end=index,
                    reason="constant condition branch marker",
                )
            )
    return findings


def coverage_summary(
    *,
    root: Path,
    py_files: list[Path],
    web_code_files: list[Path],
    python_reachability: ReachabilityResult,
    web_reachability: ReachabilityResult,
) -> dict[str, object]:
    totals = defaultdict(int)
    by_language: dict[str, dict[str, int]] = {
        "python": defaultdict(int),
        "web": defaultdict(int),
    }

    python_dead_branch_findings_all = [
        finding
        for path in py_files
        for finding in python_dead_branch_findings(path)
    ]
    reachable_python_files = python_reachability.reachable
    reachable_web_files = web_reachability.reachable

    debug_hits_preview: list[dict[str, object]] = []
    for path in [*py_files, *web_code_files]:
        metrics = line_metrics(path)
        bucket = by_language["python"] if path.suffix == ".py" else by_language["web"]
        bucket["total_lines"] += metrics.total_lines
        bucket["blank_lines"] += metrics.blank_lines
        bucket["comment_lines"] += metrics.comment_lines
        bucket["executable_lines"] += metrics.executable_lines

        totals["total_lines"] += metrics.total_lines
        totals["blank_lines"] += metrics.blank_lines
        totals["comment_lines"] += metrics.comment_lines
        totals["executable_lines"] += metrics.executable_lines

        is_orphan = (
            path.suffix == ".py" and path not in reachable_python_files
        ) or (
            path.suffix in {".ts", ".tsx", ".js", ".jsx"} and path not in reachable_web_files
        )
        if is_orphan:
            bucket["orphan_executable_lines"] += metrics.executable_lines
            totals["orphan_executable_lines"] += metrics.executable_lines

        debug_hits = debug_line_hits(path)
        bucket["debug_console_lines"] += len(debug_hits)
        totals["debug_console_lines"] += len(debug_hits)
        for line_number in debug_hits[:3]:
            if len(debug_hits_preview) >= 20:
                break
            debug_hits_preview.append(
                {
                    "path": relpath(path, root),
                    "line": line_number,
                }
            )

    python_dead_branch_lines = sum(
        finding.line_count
        for finding in python_dead_branch_findings_all
        if finding.path in reachable_python_files
    )
    totals["dead_branch_lines"] += python_dead_branch_lines
    by_language["python"]["dead_branch_lines"] += python_dead_branch_lines

    javascript_constant_branch_hits = [
        finding
        for path in web_code_files
        for finding in javascript_constant_branch_findings(path)
    ]
    by_language["web"]["constant_branch_markers"] += len(javascript_constant_branch_hits)
    totals["constant_branch_markers"] += len(javascript_constant_branch_hits)

    totals["reachable_executable_lines"] = max(
        0,
        totals["executable_lines"] - totals["orphan_executable_lines"] - totals["dead_branch_lines"],
    )
    by_language["python"]["reachable_executable_lines"] = max(
        0,
        by_language["python"]["executable_lines"] - by_language["python"]["orphan_executable_lines"] - by_language["python"]["dead_branch_lines"],
    )
    by_language["web"]["reachable_executable_lines"] = max(
        0,
        by_language["web"]["executable_lines"] - by_language["web"]["orphan_executable_lines"],
    )

    return {
        "methodology": "static",
        "automated_test_files_found": len(discover_tests(root)),
        "tests_detected": [relpath(path, root) for path in discover_tests(root)],
        "totals": dict(totals),
        "by_language": {key: dict(value) for key, value in by_language.items()},
        "debug_console_preview": debug_hits_preview,
        "dead_branch_preview": [
            {
                "path": relpath(finding.path, root),
                "line_start": finding.line_start,
                "line_end": finding.line_end,
                "reason": finding.reason,
            }
            for finding in python_dead_branch_findings_all[:20]
        ],
        "javascript_constant_branch_preview": [
            {
                "path": relpath(finding.path, root),
                "line": finding.line_start,
                "reason": finding.reason,
            }
            for finding in javascript_constant_branch_hits[:20]
        ],
    }


def build_report(root: Path) -> dict[str, object]:
    all_files = walk_repo_files(root)
    source_files = [path for path in all_files if path.suffix in SOURCE_EXTENSIONS]
    py_files = [path for path in source_files if path.suffix == ".py"]
    web_files = [
        path
        for path in source_files
        if path.suffix in WEB_EXTENSIONS and relpath(path, root).startswith("web/src/")
    ]
    legacy_frontend_files = [
        path
        for path in source_files
        if path.suffix in LEGACY_FRONTEND_EXTENSIONS and relpath(path, root).startswith("frontend/")
    ]

    python_entrypoints = [root / name for name in ("app.py", "worker.py", "pipeline.py", "task_queue.py")]
    python_entrypoints.extend(sorted((root / "scripts").glob("*.py")))
    python_module_index = build_python_module_index(py_files, root)
    python_edges = {path: python_dependencies(path, python_module_index, root) for path in py_files}
    python_reachability = compute_reachability(set(py_files), python_entrypoints, python_edges)

    web_entrypoints = [root / "web" / "src" / "main.tsx"]
    web_edges = {path: web_dependencies(path) for path in web_files}
    web_reachability = compute_reachability(set(web_files), web_entrypoints, web_edges)

    legacy_entrypoints = [
        root / "frontend" / "index.html",
        root / "frontend" / "override.html",
        root / "frontend" / "annotate.html",
    ]
    legacy_edges = {path: html_dependencies(path) if path.suffix == ".html" else set() for path in legacy_frontend_files}
    legacy_reachability = compute_reachability(set(legacy_frontend_files), legacy_entrypoints, legacy_edges)

    shell_files = [path for path in source_files if path.suffix == ".sh" or path.name in {"load.sh", "init.md"}]
    markdown_files = [path for path in source_files if path.suffix == ".md"]
    json_files = [path for path in source_files if path.suffix == ".json"]
    web_code_files = [path for path in web_files if path.suffix in CODE_EXTENSIONS]
    tests_detected = discover_tests(root)
    models = analyze_models(root)
    static_coverage = coverage_summary(
        root=root,
        py_files=py_files,
        web_code_files=web_code_files,
        python_reachability=python_reachability,
        web_reachability=web_reachability,
    )

    likely_cleanup_hotspots = sorted(
        Counter(path.parent.relative_to(root).as_posix() for path in (*python_reachability.orphan_candidates, *web_reachability.orphan_candidates, *legacy_reachability.orphan_candidates)).items(),
        key=lambda item: (-item[1], item[0]),
    )

    return {
        "repo_root": str(root),
        "summary": {
            "all_files_considered": len(all_files),
            "source_files_considered": len(source_files),
            "extension_inventory": extension_inventory(source_files),
            "top_level_inventory": directory_inventory(source_files, root),
        },
        "python": {
            "total_files": len(py_files),
            "entrypoints": [relpath(path, root) for path in sorted(path for path in python_entrypoints if path.exists())],
            "reachable_files": len(python_reachability.reachable),
            "orphan_candidate_count": len(python_reachability.orphan_candidates),
            "orphan_candidates": [relpath(path, root) for path in python_reachability.orphan_candidates],
        },
        "web": {
            "total_files": len(web_files),
            "entrypoints": [relpath(path, root) for path in web_entrypoints if path.exists()],
            "reachable_files": len(web_reachability.reachable),
            "orphan_candidate_count": len(web_reachability.orphan_candidates),
            "orphan_candidates": [relpath(path, root) for path in web_reachability.orphan_candidates],
        },
        "legacy_frontend": {
            "total_files": len(legacy_frontend_files),
            "entrypoints": [relpath(path, root) for path in legacy_entrypoints if path.exists()],
            "reachable_files": len(legacy_reachability.reachable),
            "orphan_candidate_count": len(legacy_reachability.orphan_candidates),
            "orphan_candidates": [relpath(path, root) for path in legacy_reachability.orphan_candidates],
        },
        "other_source_sets": {
            "shell_like_files": [relpath(path, root) for path in sorted(shell_files)],
            "markdown_files": [relpath(path, root) for path in sorted(markdown_files)],
            "json_files": [relpath(path, root) for path in sorted(json_files)],
        },
        "models": models,
        "static_coverage": static_coverage,
        "tests": {
            "count": len(tests_detected),
            "files": [relpath(path, root) for path in tests_detected],
        },
        "cleanup_hotspots": [
            {"directory": directory, "candidate_count": count}
            for directory, count in likely_cleanup_hotspots
        ],
        "manual_review_notes": [
            "Reachability is still file-level first, but the report now adds dead-branch heuristics for reachable Python files and constant-condition markers for reachable web code.",
            "Generated and artifact-heavy folders are intentionally excluded from dead-code scoring: outputs/, models/, logs/, dist/, .venv/, node_modules/, .git/, and cache directories.",
            "Legacy frontend files under frontend/ are still treated as live because FastAPI serves /ui, /override, and /annotate from that folder.",
            "React dead-code candidates are derived from the import graph rooted at web/src/main.tsx, so dynamically loaded files need manual review before deletion.",
            "Python dead-code candidates are derived from import reachability from app.py, worker.py, pipeline.py, task_queue.py, and scripts/*.py. Dynamic imports and reflection can still produce false positives.",
            "Static coverage is not runtime test coverage. It estimates executable, orphaned, dead-branch, and debug-oriented lines from source parsing, and it should be interpreted alongside the detected automated test count.",
            "Startup wrappers and docs are inventoried but not dead-code scored, because they are operational entrypoints rather than imported modules.",
            "Before cleanup, check for archived experiments, old evaluation artifacts, and README references so you do not remove a file that is still part of the team workflow.",
        ],
        "quick_hits": {
            "python_orphan_preview": top_candidates(python_reachability.orphan_candidates, root),
            "web_orphan_preview": top_candidates(web_reachability.orphan_candidates, root),
            "legacy_frontend_orphan_preview": top_candidates(legacy_reachability.orphan_candidates, root),
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    python = report["python"]
    web = report["web"]
    legacy = report["legacy_frontend"]
    quick_hits = report["quick_hits"]
    hotspots = report["cleanup_hotspots"]
    notes = report["manual_review_notes"]
    models = report["models"]
    static_coverage = report["static_coverage"]
    tests = report["tests"]

    def bullet_list(items: Iterable[str], empty: str) -> str:
        items = list(items)
        if not items:
            return f"- {empty}"
        return "\n".join(f"- {item}" for item in items)

    def format_number(value: object) -> str:
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def format_line_reference(item: dict[str, object]) -> str:
        line = item.get("line")
        if line is not None:
            return f"`{item['path']}`:{line}"
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        if line_start == line_end:
            return f"`{item['path']}`:{line_start}"
        return f"`{item['path']}`:{line_start}-{line_end}"

    extension_lines = "\n".join(
        f"- `{extension}`: {count}"
        for extension, count in summary["extension_inventory"].items()  # type: ignore[index]
    )
    hotspot_lines = bullet_list(
        [f"`{item['directory']}`: {item['candidate_count']} candidate files" for item in hotspots],  # type: ignore[index]
        "No hotspot directories detected.",
    )
    model_default_lines = bullet_list(
        [
            f"`{name}`: `{path}`"
            for name, path in models["runtime_defaults"].items()  # type: ignore[index]
            if path
        ],
        "No active runtime model defaults detected.",
    )
    model_file_lines = bullet_list(
        [
            f"`{item['path']}`: `{item['status']}`. {item['utilized']}"
            for item in models["files"]  # type: ignore[index]
        ],
        "No model files found under `models/`.",
    )
    coverage_totals = static_coverage["totals"]  # type: ignore[index]
    coverage_by_language = static_coverage["by_language"]  # type: ignore[index]
    coverage_total_lines = [
        f"- Total lines scanned: {format_number(coverage_totals.get('total_lines', 0))}",
        f"- Blank lines: {format_number(coverage_totals.get('blank_lines', 0))}",
        f"- Comment-only lines: {format_number(coverage_totals.get('comment_lines', 0))}",
        f"- Executable lines: {format_number(coverage_totals.get('executable_lines', 0))}",
        f"- Reachable executable lines: {format_number(coverage_totals.get('reachable_executable_lines', 0))}",
        f"- Non-reachable executable lines: {format_number(coverage_totals.get('orphan_executable_lines', 0))}",
        f"- Dead-branch executable lines: {format_number(coverage_totals.get('dead_branch_lines', 0))}",
        f"- Debug / console lines: {format_number(coverage_totals.get('debug_console_lines', 0))}",
        f"- Constant branch markers in web code: {format_number(coverage_totals.get('constant_branch_markers', 0))}",
    ]
    coverage_language_lines: list[str] = []
    for language, metrics in coverage_by_language.items():  # type: ignore[assignment]
        coverage_language_lines.extend(
            [
                f"- `{language}` total executable lines: {format_number(metrics.get('executable_lines', 0))}",
                f"- `{language}` reachable executable lines: {format_number(metrics.get('reachable_executable_lines', 0))}",
                f"- `{language}` non-reachable executable lines: {format_number(metrics.get('orphan_executable_lines', 0))}",
                f"- `{language}` debug / console lines: {format_number(metrics.get('debug_console_lines', 0))}",
            ]
        )
        if language == "python":
            coverage_language_lines.append(
                f"- `{language}` dead-branch lines: {format_number(metrics.get('dead_branch_lines', 0))}"
            )
        if language == "web":
            coverage_language_lines.append(
                f"- `{language}` constant-branch markers: {format_number(metrics.get('constant_branch_markers', 0))}"
            )

    debug_lines = bullet_list(
        [
            format_line_reference(item)
            for item in static_coverage["debug_console_preview"]  # type: ignore[index]
        ],
        "No debug or console lines detected in scanned Python/React source.",
    )
    dead_branch_lines = bullet_list(
        [
            f"{format_line_reference(item)}: {item['reason']}"
            for item in static_coverage["dead_branch_preview"]  # type: ignore[index]
        ],
        "No Python dead-branch heuristics were flagged.",
    )
    js_constant_branch_lines = bullet_list(
        [
            f"{format_line_reference(item)}: {item['reason']}"
            for item in static_coverage["javascript_constant_branch_preview"]  # type: ignore[index]
        ],
        "No constant-condition web branches were flagged.",
    )
    test_lines = bullet_list(
        [f"`{item}`" for item in tests["files"]],  # type: ignore[index]
        "No automated test files detected by the current patterns.",
    )

    return "\n".join(
        [
            "# Repo Diagnostics",
            "",
            f"Repo root: `{report['repo_root']}`",
            "",
            "## Summary",
            f"- Files considered: {summary['all_files_considered']}",  # type: ignore[index]
            f"- Source files considered: {summary['source_files_considered']}",  # type: ignore[index]
            "",
            "### Extensions",
            extension_lines or "- No source files found.",
            "",
            "## Python Reachability",
            f"- Total Python files: {python['total_files']}",  # type: ignore[index]
            f"- Reachable from app/worker/pipeline/scripts: {python['reachable_files']}",  # type: ignore[index]
            f"- Orphan candidates: {python['orphan_candidate_count']}",  # type: ignore[index]
            "### Preview",
            bullet_list(quick_hits["python_orphan_preview"], "No Python orphan candidates."),  # type: ignore[index]
            "",
            "## React Reachability",
            f"- Total React/Vite files: {web['total_files']}",  # type: ignore[index]
            f"- Reachable from web/src/main.tsx: {web['reachable_files']}",  # type: ignore[index]
            f"- Orphan candidates: {web['orphan_candidate_count']}",  # type: ignore[index]
            "### Preview",
            bullet_list(quick_hits["web_orphan_preview"], "No React orphan candidates."),  # type: ignore[index]
            "",
            "## Legacy Frontend Reachability",
            f"- Total legacy frontend files: {legacy['total_files']}",  # type: ignore[index]
            f"- Reachable from legacy HTML entrypoints: {legacy['reachable_files']}",  # type: ignore[index]
            f"- Orphan candidates: {legacy['orphan_candidate_count']}",  # type: ignore[index]
            "### Preview",
            bullet_list(quick_hits["legacy_frontend_orphan_preview"], "No legacy frontend orphan candidates."),  # type: ignore[index]
            "",
            "## Model Files In Play",
            "### Runtime Defaults",
            model_default_lines,
            "",
            "### Model Inventory",
            model_file_lines,
            "",
            "### Notes",
            bullet_list(models["notes"], "No model notes."),  # type: ignore[index]
            "",
            "## Static Coverage",
            f"- Methodology: `{static_coverage['methodology']}`",  # type: ignore[index]
            f"- Automated test files detected: {tests['count']}",  # type: ignore[index]
            "### Totals",
            "\n".join(coverage_total_lines),
            "",
            "### By Language",
            "\n".join(coverage_language_lines) or "- No language metrics available.",
            "",
            "### Detected Test Files",
            test_lines,
            "",
            "### Debug / Console Preview",
            debug_lines,
            "",
            "## Dead Branch Heuristics",
            "### Python Findings",
            dead_branch_lines,
            "",
            "### Web Constant-Condition Markers",
            js_constant_branch_lines,
            "",
            "## Cleanup Hotspots",
            hotspot_lines,
            "",
            "## Manual Review Notes",
            bullet_list(notes, "No extra notes."),
            "",
        ]
    )


def write_report(report: dict[str, object], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), "utf-8")
    md_out.write_text(render_markdown(report), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan the repo for dead-file candidates, dead-branch heuristics, static coverage estimates, and model-file utilization."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repo root to scan.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON, help="Where to write the JSON report.")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD, help="Where to write the markdown report.")
    parser.add_argument("--stdout", action="store_true", help="Print the markdown report to stdout.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    report = build_report(root)
    write_report(report, args.json_out.expanduser().resolve(), args.md_out.expanduser().resolve())

    print(f"Wrote JSON report to {args.json_out.expanduser().resolve()}")
    print(f"Wrote Markdown report to {args.md_out.expanduser().resolve()}")
    print(
        "Dead-code candidates are heuristics only. Review manual_review_notes and entrypoints before deleting files."
    )
    if args.stdout:
        print()
        print(render_markdown(report))


if __name__ == "__main__":
    main()
