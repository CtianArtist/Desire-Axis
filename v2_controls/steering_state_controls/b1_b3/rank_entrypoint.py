"""torchrun child wrapper: diagnose even imports before the production entrypoint."""
import os
import runpy
import sys
import traceback

from pain_axis_b import diagnostics as diag


def main():
    output = os.environ["PAIN_OUTPUT"]
    code = 1
    try:
        diag.stage(output, "rank_entrypoint", "before")
        script, *args = sys.argv[1:]
        sys.argv = [script, *args]
        # Protocol module imports happen inside this try, after torchrun's stream redirects.
        runpy.run_path(script, run_name="__main__")
        code = 0
        diag.stage(output, "rank_entrypoint", "passed")
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            code = 0
        else:
            code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
            try:
                diag.record_exception(exc, output)
            except BaseException:
                # Never hide the original failure if saving its receipt also fails.
                try:
                    traceback.print_exc()
                except BaseException:
                    pass
            raise
    finally:
        try:
            diag.record_exit(code, output)
        except BaseException:
            try:
                traceback.print_exc()
            except BaseException:
                pass
            if code == 0:
                raise


if __name__ == "__main__":
    main()
