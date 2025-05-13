import sys
import os
import importlib


def init_notebook_path():
    # Add the project root to Python path if needed
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Reload app module if it was partially imported
    if "app" in sys.modules:
        importlib.reload(sys.modules["app"])

    print(f"Python path properly configured. Project root added: {project_root}")

    return project_root  # Return the path for use with Ray
