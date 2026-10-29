"""Original launch command, now opening the shared AI classification tabs."""
import sys
from local_tagging_server import main as launch


def main(argv=None):
    return launch(['--port', '5000', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    raise SystemExit(main())
