# Release process

This checklist is for maintainers preparing a tagged release of
`j48-python-backend`. Run commands from the repository root and release only
from a clean commit on the default branch.

## 1. Choose the version

The project follows semantic versioning. While the version is `0.x`, minor
releases may change public behavior.

Update the version in both locations:

- `pyproject.toml` (`project.version`), and
- `j48/__init__.py` (`__version__`).

Add a dated entry to `CHANGELOG.md`. Use a PEP 440 version such as `0.4.1` for
a final release or `0.5.0rc1` for a release candidate. The corresponding Git
tags are `v0.4.1` and `v0.5.0rc1`.

Verify that the two declarations agree:

```bash
python - <<'PY'
import ast
import tomllib
from pathlib import Path

with open("pyproject.toml", "rb") as stream:
    project_version = tomllib.load(stream)["project"]["version"]

module = ast.parse(Path("j48/__init__.py").read_text())
module_version = next(
    node.value.value
    for node in module.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "__version__"
        for target in node.targets
    )
    and isinstance(node.value, ast.Constant)
    and isinstance(node.value.value, str)
)
assert module_version == project_version, (module_version, project_version)
print(project_version)
PY
```

This check deliberately reads the module without importing it, so it also
works before the project's runtime dependencies are installed. The package
workflow additionally requires a matching `CHANGELOG.md` section, and on tag
builds requires the tag to be exactly `v<project.version>`.

## 2. Run the full test matrix

Create a clean virtual environment and install all development dependencies:

```bash
python -m venv .venv-release
. .venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
python -m pytest -ra
```

The GitHub Actions test workflow is the source of truth for supported Python
versions. It also exercises the NumPy fallback without numba and the oldest
supported dependency versions. Every job must pass on the release commit.

Run the WEKA parity suite separately with Java and WEKA 3.8.6 available:

```bash
base=https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka
curl -sSfLO "$base/weka-stable/3.8.6/weka-stable-3.8.6.jar"
curl -sSfLO "$base/thirdparty/bounce/0.18/bounce-0.18.jar"
J48_WEKA_CLASSPATH="$PWD/weka-stable-3.8.6.jar:$PWD/bounce-0.18.jar" \
  python -m pytest -ra -m weka tests/test_weka_parity.py
```

Only the strict expected failures documented in `tests/test_weka_parity.py`
are acceptable. Review any new skip, failure or unexpected pass before
continuing.

## 3. Build and inspect artifacts

Remove outputs from earlier builds, then create both the source distribution
and wheel:

```bash
rm -rf build dist
python -m pip install --upgrade build twine
python -m build
python -m twine check --strict dist/*
```

The `package` GitHub Actions workflow performs the same build and metadata
check, installs and smoke-tests both distributions, generates SHA-256
checksums and uploads the artifacts for inspection.

Test the wheel from outside the checkout so that Python cannot import the
source tree accidentally:

```bash
python -m venv /tmp/j48-release-venv
/tmp/j48-release-venv/bin/python -m pip install dist/*.whl
(
  cd /tmp
  /tmp/j48-release-venv/bin/python - <<'PY'
import numpy as np

from j48 import J48Classifier, J48FastClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 1, 1])
for estimator_type in (J48Classifier, J48FastClassifier):
    estimator = estimator_type().fit(X, y)
    assert estimator.predict(X).tolist() == y.tolist()
print("wheel smoke test: OK")
PY
)
```

Inspect the contents of both artifacts before publishing:

```bash
tar -tzf dist/*.tar.gz
python -m zipfile --list dist/*.whl
```

Generate checksums for the files that will be attached to the release:

```bash
(cd dist && sha256sum -- *.whl *.tar.gz > SHA256SUMS)
(cd dist && sha256sum --check SHA256SUMS)
```

## 4. Tag and publish

Confirm the checkout is clean and the release commit has passed CI:

```bash
git status --short
git log -1 --oneline
```

Create an annotated tag whose version exactly matches the package metadata:

```bash
version=$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
git tag -a "v$version" -m "j48-python-backend $version"
git push origin "v$version"
```

Pushing the tag starts the `package` workflow. After the build, metadata and
wheel smoke test pass, its `github-release` job creates a draft GitHub Release
and attaches the wheel, source distribution and `SHA256SUMS`. Review the
automatically generated notes against the matching `CHANGELOG.md` section,
verify the checksums, edit the notes if necessary, and publish the draft
manually.

If the package is published to PyPI, use a protected GitHub environment and
PyPI Trusted Publishing rather than a long-lived API token. Test the publishing
workflow against TestPyPI before enabling production publishing.

## 5. Verify the published release

In another clean environment, install the exact released version and repeat
the smoke test:

```bash
python -m venv /tmp/j48-published-venv
/tmp/j48-published-venv/bin/python -m pip install \
  "j48-python-backend==$version"
```

Check that the GitHub Release and package index show the expected README,
license, Python requirement, version and artifacts. Finally, open the next
development section in `CHANGELOG.md` if more changes are planned.
