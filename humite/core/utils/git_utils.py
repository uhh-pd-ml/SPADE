import subprocess  # nosec
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


def run_git_command_capture_output(command: Sequence[str], cwd: Optional[str] = None) -> str:
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, cwd=cwd)
        return output.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def collect_git_metadata(cwd: Optional[str] = None) -> Dict[str, str]:
    return {
        "commit": run_git_command_capture_output(["git", "rev-parse", "HEAD"], cwd=cwd),
        "branch": run_git_command_capture_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd
        ),
        "diff": run_git_command_capture_output(["git", "diff"], cwd=cwd),
    }


def write_git_diff_file(
    git_metadata: Dict[str, str], destination_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    diff_text = git_metadata.get("diff", "")
    if not diff_text:
        return None, None
    commit = git_metadata.get("commit", "")
    diff_name = f"git-diff_{commit[:7]}.patch" if commit else "git-diff.patch"
    diff_path = Path(destination_dir) / diff_name
    diff_path.write_text(diff_text, encoding="utf-8")
    return str(diff_path), diff_text[:2000]


def get_git_hash() -> str:
    return run_git_command_capture_output(["git", "rev-parse", "HEAD"])


def get_git_status() -> str:
    command = "git diff -- . ':!*.ipynb' --color"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    git_diff_output = stdout.decode("utf-8")
    separator_start = f"\n{100 * '='}\n{'=' * 10} start git diff {'=' * 10}\n"
    separator_end = f"\n{'=' * 10} end git diff {'=' * 10}\n{100 * '='}\n"
    return separator_start + git_diff_output + separator_end


def get_last_commit_message() -> str:
    return run_git_command_capture_output(["git", "log", "-1", "--pretty=%B"])
