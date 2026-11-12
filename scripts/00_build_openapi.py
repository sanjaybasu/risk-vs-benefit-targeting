"""Write contract/openapi.yaml from the pydantic models in src/system_three/contract.py."""

from pathlib import Path

import yaml

from system_three.contract import build_openapi

OUT = Path(__file__).resolve().parents[1] / "contract" / "openapi.yaml"


def render() -> str:
    return yaml.safe_dump(build_openapi(), sort_keys=False, allow_unicode=True, width=100)


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
