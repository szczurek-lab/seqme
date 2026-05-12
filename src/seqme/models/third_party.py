import pickle
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


class ThirdPartyModel:
    """Wrapper for loading and calling a third-party plugin model.

    Clones a Git repository into a local directory (if not already present) and
    invokes a specified function from that repository in an isolated uv environment.

    uv is used over conda because it provides precise lockfile-based reproducibility,
    strict package version pinning, and per-project Python version isolation — ensuring
    the plugin runs in exactly the environment its author intended.

    Officially supported models can be found here: https://github.com/szczurek-lab/seqme-thirdparty

    Note:
        ``uv`` and ``git`` may not be on the ``PATH`` of a Jupyter kernel. In that
        case, pass their absolute paths via the ``uv`` and ``git`` parameters.

    Examples:
        >>> import seqme as sm
        >>> hello_model = sm.models.ThirdPartyModel(
        ...     entry_point="hello_model.model:embed",
        ...     path="./thirdparty/hello-model",
        ...     url="https://github.com/szczurek-lab/seqme-thirdparty",
        ...     branch="main",
        ... )
        >>> hello_model(["MKQW", "RKSPL"], batch_size=32)
        array([[44.,  8., 12., 32.],
               [55., 10., 15., 40.]])
    """

    def __init__(
        self,
        entry_point: str,
        path: str | Path,
        url: str | None = None,
        branch: str | None = None,
        shallow: bool = True,
        extras: list[str] | None = None,
        uv: str | Path | None = None,
        git: str | Path | None = None,
    ):
        """
        Initialize the third-party model.

        Args:
            entry_point: Module and function to call, in the form 'module.path:function'.
            path: Path to the plugin repository. Cloned here if it does not exist and url is provided.
            url: Git repository URL to clone (optionally prefixed with 'git+'). If None, path must already exist.
            branch: Branch to clone. If None, clones the default branch.
            shallow: If True, clones only the latest commit (no full history). Defaults to True.
            extras: Optional dependency groups from the project to install, e.g. ``['cpu', 'cuda']``.
                Each entry is passed to ``uv sync`` via ``--extra``.
            uv: Path to the uv executable. If None, 'uv' is looked up on PATH.
            git: Path to the git executable. If None, 'git' is looked up on PATH.

        Raises:
            ValueError: If entry_point is not of the form 'module:function'.
            FileNotFoundError: If path does not exist and no url is provided.
        """
        try:
            module, fn = entry_point.split(":", 1)
        except ValueError as e:
            raise ValueError("entry_point must be of the form 'module:function'.") from e

        if not module or not fn:
            raise ValueError("entry_point must be of the form 'module:function'.")

        self.repo_dir = Path(path).resolve()
        self.module = module
        self.fn = fn
        self.extras = extras or []
        self.uv = str(uv) if uv is not None else "uv"
        self.git = str(git) if git is not None else "git"

        _check_tool(self.uv)

        if not self.repo_dir.exists():
            if url is None:
                raise FileNotFoundError(f"'{self.repo_dir}' does not exist. Provide a url to clone it.")

            _check_tool(self.git)
            _clone_git_repository(self.repo_dir, url, branch, shallow, self.git)

        _sync(self.repo_dir, self.extras, self.uv)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute the plugin's function with the given arguments.

        Args:
            *args: Positional arguments passed to the plugin function.
            **kwargs: Keyword arguments passed to the plugin function.

        Returns:
            The return value of the plugin function.
        """
        return _run(self.repo_dir, self.module, self.fn, args, kwargs, self.uv)


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' is not installed or not found on PATH.")


def _clone_git_repository(
    repo_dir: Path,
    repo_url: str,
    branch: str | None = None,
    shallow: bool = True,
    git: str = "git",
):
    if repo_dir.exists():
        raise FileExistsError(f"'{repo_dir}' already exists.")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    url = repo_url.removeprefix("git+")
    clone_cmd = [git, "clone", url, str(repo_dir)]
    if branch:
        clone_cmd += ["-b", branch, "--single-branch"]
    if shallow:
        clone_cmd += ["--depth", "1"]

    try:
        subprocess.run(clone_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed:\n{e.stderr}") from e


def _sync(repo_dir: Path, extras: list[str], uv: str = "uv") -> None:
    extra_flags = [flag for extra in extras for flag in ("--extra", extra)]
    try:
        subprocess.run(
            [uv, "sync", "--project", str(repo_dir), *extra_flags],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"uv sync failed:\n{e.stderr}") from e


def _wrap_code(module: str, fn: str) -> str:
    return (
        "import pickle, sys;"
        f"import {module};"
        "args, kwargs = pickle.load(open(sys.argv[1],'rb'));"
        f"result = {module}.{fn}(*args, **kwargs);"
        "pickle.dump(result, open(sys.argv[2],'wb'))"
    )


def _run(repo_dir: Path, module: str, fn: str, args: tuple, kwargs: dict, uv: str = "uv") -> Any:
    with TemporaryDirectory(prefix="seqme") as tmpdir:
        in_path = Path(tmpdir) / "input.pkl"
        out_path = Path(tmpdir) / "output.pkl"

        with open(in_path, "wb") as f:
            pickle.dump((args, kwargs), f)

        code = _wrap_code(module, fn)
        cmd = [uv, "run", "--no-sync", "--project", str(repo_dir), "python", "-c", code, str(in_path), str(out_path)]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Plugin subprocess failed:\n{e.stderr}") from e

        with open(out_path, "rb") as f:
            return pickle.load(f)
