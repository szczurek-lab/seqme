import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import numpy as np


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
        repo_url: str,
        save_dir: str,
        python_bin: Optional[str] = None,
    ):
        """
        Initialize and install the plugin.

        Args:
            entry_point (str): String specifying the module and function,
                in the form 'module.path:func_name'.
            repo_url (str): Git repository URL for the plugin (prefixed with 'git+' or local path).
            save_dir (str): Directory where the virtual environment and repo will be stored.
            python_bin: Optional path to a python executable. If None, creates a venv enviroment using the exposed python executable.

        Raises:
            ValueError: If entry_point is not of the form 'module:func'.
        """
        try:
            module, func = entry_point.split(":", 1)
        except ValueError:
            raise ValueError("entry_point must be of the form 'module:func'")

        self.module = module
        self.func = func

        self.plugin = Plugin()
        self.plugin.setup(
            plugins_root=Path(save_dir),
            repo_url=repo_url,
            python_bin=python_bin,
        )

    def __call__(self, sequences: list[str], **kwargs) -> np.ndarray:
        """
        Execute the plugin's function with the given keyword arguments.

        Args:
            sequences: List of sequences.
            **kwargs: Arbitrary keyword arguments to pass to the plugin function.

        Returns:
            np.ndarray: The result returned by the plugin function.

        Raises:
            ValueError: If the plugin response is not a numpy.ndarray.
        """
        kwargs["sequences"] = sequences
        result = self.plugin.run(self.module, self.func, kwargs)
        if not isinstance(result, np.ndarray):
            raise ValueError("Invalid plugin response: expected numpy.ndarray")
        return result


class Plugin:
    """
    Manages the environment and execution of a third-party plugin.
    """

    def setup(
        self,
        plugins_root: Path,
        repo_url: str,
        python_bin: Optional[str] = None,
    ):
        """
        Create a virtual environment and install the plugin from its repository.

        Args:
            plugins_root (Path): Root directory for plugin environments and repos.
            repo_url (str): Git repository URL for the plugin.
            python_bin: Optional path to a python executable. If None, creates a venv enviroment using the exposed python executable.
        """
        plugins_root.mkdir(parents=True, exist_ok=True)
        repo_dir = plugins_root / "repo"

        if python_bin is None:
            env_dir = plugins_root / "env"
            python_bin = env_dir / "bin" / "python"

            if not env_dir.exists():
                subprocess.check_call([sys.executable, "-m", "venv", str(env_dir)])
                subprocess.check_call(
                    [str(python_bin), "-m", "pip", "install", "--upgrade", "pip"]
                )

        if not repo_dir.exists():
            url = repo_url.removeprefix("git+")
            subprocess.check_call(["git", "clone", url, str(repo_dir)])
            subprocess.check_call(
                [str(python_bin), "-m", "pip", "install", "-e", str(repo_dir)]
            )

        self.python_bin = python_bin

    def run(
        self,
        module: str,
        func: str,
        arguments: dict,
    ) -> Any:
        """
        Run the specified function from the installed plugin.

        Args:
            module (str): The module path containing the function to call.
            func (str): The name of the function within the module.
            arguments (dict): Dictionary of arguments passed to the function.

        Returns:
            Any: The deserialized output of the plugin's function.
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
                f"result = {module}.{func}(**data);"
                "pickle.dump(result, open(sys.argv[2],'wb'))"
            )
            cmd = [str(self.python_bin), "-c", code, str(in_path), str(out_path)]
            subprocess.check_call(cmd)

            # Deserialize and return output
            with open(out_path, "rb") as f:
                return pickle.load(f)
