import builtins
import inspect
import json
import sys
from typing import Any


def main() -> None:
    # Make verifier_utils importable from the same directory as this worker,
    # regardless of how the subprocess was launched (PYTHONPATH-free).
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    try:
        payload = json.load(sys.stdin)
        func_src: str = payload["func_src"]
        args: dict[str, Any] = payload.get("args", {})
        func_name: str = payload.get("func_name", "evaluate")

        # Give user code full access to Python builtins (HIGH RISK).
        # This includes `__import__`, so `import ...` inside func_src will work.
        g: dict[str, Any] = {"__builtins__": builtins}

        code = compile(func_src, filename="<user_function>", mode="exec")
        exec(code, g, g)

        if func_name not in g or not callable(g[func_name]):
            raise ValueError(f"Function '{func_name}' not found after compiling source.")

        fn = g[func_name]

        # Filter kwargs to the function signature
        sig = inspect.signature(fn)
        filtered = {k: v for k, v in args.items() if k in sig.parameters}
        sig.bind(**filtered)  # raises if required args are missing

        result = fn(**filtered)

        sys.stdout.write(f"<answer>{result}</answer>")
        sys.stdout.flush()

    except Exception as e:
        result = False
        sys.stdout.write(f"<answer>{result}</answer>")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
