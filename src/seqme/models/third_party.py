import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


class ThirdPartyModel:
    """
    Wrapper for loading and calling a third-party plugin model.

    This class handles installation of the plugin into a dedicated virtual
    environment and provides a __call__ interface to invoke the plugin's
    entry point function.
    """

    def __init__(
        self,
        entry_point: str,
        repo_path: str,
        python_bin: str | None = None,
        repo_url: str | None = None,
        branch: str | None = None,
    ):
        """
        Initialize and install the plugin.

        Args:
            entry_point: String specifying the module and function, in the form 'module.path:function'.
            repo_path: Root directory for plugin environments and repos.
            python_bin: Optional path to a python executable. If ``None``, creates a venv enviroment using the exposed python executable.
            repo_url: Optional Git repository URL for the plugin (Optionally prefixed with 'git+').
            branch: Optional branch to clone. If ``None``, clone the whole repository.

        Raises:
            ValueError: If entry_point is not of the form 'module:function'.
        """
        try:
            module, fn = entry_point.split(":", 1)
        except ValueError as e:
            raise ValueError("entry_point must be of the form 'module:function'.") from e

        self.module = module
        self.fn = fn

        self.plugin = Plugin()
        self.plugin.setup(
            plugins_root=Path(repo_path),
            repo_url=repo_url,
            branch=branch,
            python_bin=Path(python_bin) if python_bin else None,
        )

    def __call__(self, sequences: list[str], **kwargs) -> Any:
        """
        Execute the plugin's function with the given keyword arguments.

        Args:
            sequences: List of sequences.
            **kwargs: Arbitrary keyword arguments to pass to the plugin function.

        Returns:
            Model output.
        """
        kwargs["sequences"] = sequences
        return self.plugin.run(self.module, self.fn, kwargs)


class Plugin:
    """Manages the environment and execution of a third-party plugin."""

    def setup(
        self,
        plugins_root: Path,
        python_bin: Path | None,
        repo_url: str | None,
        branch: str | None,
    ):
        """
        Create a virtual environment and install the plugin from its repository.

        Args:
            plugins_root: Root directory for plugin environments and repos.
            python_bin: Optional path to a python executable. If ``None``, creates a venv enviroment using the exposed python executable.
            repo_url: Optional Git repository URL for the plugin to clone.
            branch: Optional branch to clone. If ``None``, clone the whole repository.
        """
        plugins_root.mkdir(parents=True, exist_ok=True)
        repo_dir = plugins_root / "repo"

        if python_bin is None:
            env_dir = plugins_root / "env"
            python_bin = env_dir / "bin" / "python"

            if not env_dir.exists():
                subprocess.check_call([sys.executable, "-m", "venv", str(env_dir)])
                subprocess.check_call([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
        else:
            if not python_bin.is_file():
                raise ValueError(f"Python executable not found: '{python_bin!r}'.")

        if not repo_dir.exists():
            if repo_url is None:
                raise ValueError(f"'{repo_dir}' does not exist. Correct the path or define a repository url to clone.")

            url = repo_url.removeprefix("git+")

            clone_cmd = ["git", "clone", url, str(repo_dir)]
            if branch:
                clone_cmd += ["-b", branch, "--single-branch"]

            subprocess.check_call(clone_cmd)
            subprocess.check_call([str(python_bin), "-m", "pip", "install", "-e", str(repo_dir)])

        self.python_bin = python_bin

    def run(
        self,
        module: str,
        fn: str,
        arguments: dict,
    ) -> Any:
        """
        Run the specified function from the installed plugin.

        Args:
            module: The module path containing the function to call.
            fn: The name of the function within the module.
            arguments: Dictionary of arguments passed to the function.

        Returns:
            The deserialized output of the plugin's function.
        """
        with TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "input.pkl"
            out_path = Path(tmpdir) / "output.pkl"

            # Serialize inputs
            with open(in_path, "wb") as f:
                pickle.dump(arguments, f)

            # Construct and run inline Python execution
            code = (
                "import pickle, sys;"
                f"import {module};"
                "data = pickle.load(open(sys.argv[1],'rb'));"
                f"result = {module}.{fn}(**data);"
                "pickle.dump(result, open(sys.argv[2],'wb'))"
            )
            cmd = [str(self.python_bin), "-c", code, str(in_path), str(out_path)]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("🔴 Subprocess failed:")
                print(e.stderr)
                raise

            # Deserialize and return output
            with open(out_path, "rb") as f:
                return pickle.load(f)
