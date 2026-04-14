import importlib.util
import inspect
import sys
from pathlib import Path

def import_asp_script(module_name, alias="*"):
    repo_root = Path(__file__).resolve().parent.parent

    direct_script = repo_root / f"{module_name}.py"
    nested_script = repo_root / module_name / f"{module_name}.py"
    script_path = direct_script if direct_script.is_file() else nested_script

    if not script_path.is_file():
        matches = [p for p in repo_root.rglob("*.py") if p.stem == module_name]
        if not matches:
            raise FileNotFoundError(f"No script found for module '{module_name}'")
        script_path = matches[0]

    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    namespace = caller.f_globals if caller else globals()
    del frame

    sys_module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(sys_module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[sys_module_name] = module

    if alias == "*":
        export_names = getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")])
        for name in export_names:
            namespace[name] = getattr(module, name)
    else:
        namespace[alias] = module

    return module