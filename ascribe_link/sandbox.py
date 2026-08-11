"""Firejail-based sandboxing for code execution.

Provides secure execution of untrusted Python code with:
- Filesystem isolation (private tmpfs)
- Network isolation (no network access)
- Resource limits (memory, CPU time, wall time)
- Capability dropping
- Seccomp filtering
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    memory_limit_mb: int = 4096  # 4GB
    cpu_time_limit_secs: int = 300  # 5 minutes CPU time
    wall_time_limit_secs: int = 600  # 10 minutes wall time
    network: bool = False  # No network by default
    allow_paths: list[str] | None = None  # Additional read-only paths to expose


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    output: Any | None = None  # Parsed JSON output if available
    error: str | None = None


def is_firejail_available() -> bool:
    """Check if Firejail is installed and accessible."""
    return shutil.which("firejail") is not None


def get_firejail_version() -> str | None:
    """Get Firejail version string."""
    try:
        result = subprocess.run(
            ["firejail", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # First line is usually "firejail version X.Y.Z"
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def build_firejail_command(
    command: list[str],
    working_dir: Path,
    config: SandboxConfig,
    input_files: list[Path] | None = None,
) -> list[str]:
    """Build a Firejail command with appropriate sandboxing.

    Parameters
    ----------
    command : list[str]
        The command to run (e.g., ["python3", "script.py"])
    working_dir : Path
        Working directory (will be mounted read-write)
    config : SandboxConfig
        Sandbox configuration
    input_files : list[Path], optional
        Additional files to mount read-only

    Returns
    -------
    list[str]
        Full Firejail command
    """
    # Convert wall time to HH:MM:SS format
    hours = config.wall_time_limit_secs // 3600
    minutes = (config.wall_time_limit_secs % 3600) // 60
    seconds = config.wall_time_limit_secs % 60
    timeout_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    cmd = [
        "firejail",
        "--quiet",
        # Filesystem isolation
        f"--private={working_dir}",
        "--private-tmp",
        "--private-dev",
        # Network isolation
        *([] if config.network else ["--net=none"]),
        # Privilege restrictions
        "--noroot",
        "--caps.drop=all",
        "--seccomp",
        # Resource limits
        f"--rlimit-as={config.memory_limit_mb * 1024 * 1024}",  # bytes
        f"--rlimit-cpu={config.cpu_time_limit_secs}",
        f"--timeout={timeout_str}",
        # No new privileges
        "--nonewprivs",
        # Disable some features we don't need
        "--nodvd",
        "--nogroups",
        "--nosound",
        "--notv",
        "--nou2f",
        "--novideo",
    ]

    # Add read-only bindings for input files
    if input_files:
        for f in input_files:
            if f.exists():
                # Mount to same path in sandbox
                cmd.extend(["--read-only", str(f)])

    # Add any custom allowed paths
    if config.allow_paths:
        for p in config.allow_paths:
            cmd.extend(["--read-only", p])

    # Add the actual command
    cmd.extend(["--", *command])

    return cmd


async def run_sandboxed(
    code: str,
    working_dir: Path | None = None,
    config: SandboxConfig | None = None,
    input_files: list[Path] | None = None,
    python_executable: str = "python3",
) -> SandboxResult:
    """Run Python code in a Firejail sandbox.

    Parameters
    ----------
    code : str
        Python code to execute
    working_dir : Path, optional
        Working directory. If None, creates a temp directory.
    config : SandboxConfig, optional
        Sandbox configuration. Uses defaults if None.
    input_files : list[Path], optional
        Additional files to make available (read-only)
    python_executable : str
        Python interpreter to use

    Returns
    -------
    SandboxResult
        Execution result
    """
    import tempfile

    if not is_firejail_available():
        logger.warning("Firejail not available, running without sandbox")
        return await _run_unsandboxed(code, working_dir, python_executable)

    config = config or SandboxConfig()
    cleanup_dir = False

    if working_dir is None:
        working_dir = Path(tempfile.mkdtemp(prefix="ascribe_sandbox_"))
        cleanup_dir = True

    try:
        # Write the script
        script_path = working_dir / "script.py"
        script_path.write_text(code)

        # Write a wrapper that captures output as JSON
        wrapper_code = '''
import json
import sys
import traceback
from pathlib import Path

result = {"success": False, "output": None, "error": None}

try:
    # Execute the user script
    exec(compile(Path("script.py").read_text(), "script.py", "exec"), {"__name__": "__main__"})
    
    # Check for result file
    result_file = Path("result.json")
    if result_file.exists():
        result["output"] = json.loads(result_file.read_text())
        result["success"] = True
    else:
        result["success"] = True
        result["output"] = None
        
except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}\\n{traceback.format_exc()}"

print("__SANDBOX_RESULT__")
print(json.dumps(result))
'''
        wrapper_path = working_dir / "wrapper.py"
        wrapper_path.write_text(wrapper_code)

        # Copy input files to working directory if needed
        copied_files = []
        if input_files:
            for f in input_files:
                if f.exists():
                    dest = working_dir / f.name
                    if f.is_file():
                        shutil.copy2(f, dest)
                        copied_files.append(dest)

        # Build command
        cmd = build_firejail_command(
            command=[python_executable, "wrapper.py"],
            working_dir=working_dir,
            config=config,
        )

        logger.debug("Running sandboxed command: %s", " ".join(cmd))

        # Run with timeout
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=config.wall_time_limit_secs + 30,  # Extra buffer
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

        except TimeoutError:
            proc.kill()
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error="Sandbox execution timed out",
            )

        # Parse result
        if "__SANDBOX_RESULT__" in stdout:
            parts = stdout.split("__SANDBOX_RESULT__", 1)
            pre_output = parts[0]
            try:
                result_json = json.loads(parts[1].strip())
                return SandboxResult(
                    success=result_json.get("success", False),
                    stdout=pre_output,
                    stderr=stderr,
                    return_code=proc.returncode,
                    output=result_json.get("output"),
                    error=result_json.get("error"),
                )
            except json.JSONDecodeError as e:
                return SandboxResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=proc.returncode,
                    error=f"Failed to parse sandbox result: {e}",
                )
        else:
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=proc.returncode,
                error=stderr if proc.returncode != 0 else None,
            )

    finally:
        if cleanup_dir and working_dir.exists():
            shutil.rmtree(working_dir, ignore_errors=True)


async def _run_unsandboxed(
    code: str,
    working_dir: Path | None,
    python_executable: str,
) -> SandboxResult:
    """Fallback: run without sandboxing (when Firejail unavailable)."""
    import tempfile

    cleanup_dir = False
    if working_dir is None:
        working_dir = Path(tempfile.mkdtemp(prefix="ascribe_nosandbox_"))
        cleanup_dir = True

    try:
        script_path = working_dir / "script.py"
        script_path.write_text(code)

        proc = await asyncio.create_subprocess_exec(
            python_executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=300,
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Check for result file
        result_file = working_dir / "result.json"
        output = None
        if result_file.exists():
            try:
                output = json.loads(result_file.read_text())
            except json.JSONDecodeError:
                pass

        return SandboxResult(
            success=proc.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            return_code=proc.returncode,
            output=output,
            error=stderr if proc.returncode != 0 else None,
        )

    finally:
        if cleanup_dir and working_dir.exists():
            shutil.rmtree(working_dir, ignore_errors=True)


async def run_sandboxed_mesh_generation(
    code: str,
    config: SandboxConfig | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Run mesh generation code in sandbox and extract result.

    The code should write a JSON file to 'result.json' with:
    {"vertices": [[x,y,z], ...], "indices": [i1, i2, i3, ...]}

    Parameters
    ----------
    code : str
        Python code that generates a mesh
    config : SandboxConfig, optional
        Sandbox configuration

    Returns
    -------
    tuple[list[list[float]], list[int]]
        (vertices, indices)

    Raises
    ------
    ValueError
        If code fails or doesn't produce valid mesh data
    """
    result = await run_sandboxed(code, config=config)

    if not result.success:
        raise ValueError(f"Sandbox execution failed: {result.error or result.stderr}")

    if result.output is None:
        raise ValueError(
            "Code did not produce result.json. "
            "Your code must write: json.dump({'vertices': [...], 'indices': [...]}, open('result.json', 'w'))"
        )

    vertices = result.output.get("vertices")
    indices = result.output.get("indices")

    if not vertices or not indices:
        raise ValueError(
            f"Invalid mesh result. Got vertices={type(vertices)}, indices={type(indices)}"
        )

    return vertices, indices
