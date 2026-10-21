# Releasing

Two one-time setups are needed before the first release. Both require the owner's
accounts, so an agent cannot do them.

## 1. PyPI trusted publishing (one time)

Trusted publishing lets GitHub Actions upload to PyPI without any token being stored in
this repository. On <https://pypi.org/manage/account/publishing/>, add a pending publisher:

| Field | Value |
|---|---|
| PyPI project name | `bioelectrics` |
| Owner | `tlcdv` |
| Repository name | `bioelectrics` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then, in this repository's settings, create an environment called `pypi`. No secret is
needed in it.

## 2. Zenodo (one time)

A DOI makes the library citable. On <https://zenodo.org/account/settings/github/>, sign in
with GitHub and switch on `tlcdv/bioelectrics`. Zenodo then mints a DOI for every release
from that point on.

After the first release, add the DOI badge to `README.md` and the DOI to `CITATION.cff`.

## Making a release

1. Check that `main` is green on both workflows.
2. Bump the version in `pyproject.toml`, `src/bioelectrics/__init__.py` and
   `CITATION.cff`. All three must agree.
3. Regenerate the generated artifacts and confirm they still hold:
   ```bash
   python scripts/build_validation.py      # exits non-zero if a check falls outside tolerance
   python -m pytest -q
   python -m pytest --nbmake notebooks/ -q
   ```
4. Commit, then create a GitHub release with the tag `vX.Y.Z`. The tag and the package
   version must match; `publish.yml` checks this and fails the release if they differ.
5. Publishing to PyPI and minting the DOI then happen automatically.

## What is generated, not written

- `docs/validation.md` and its figures come from `scripts/build_validation.py`.
- `assets/drug_screen.*` come from `examples/drug_repurposing.py`, which caches its results
  so `--reuse` redraws without repeating the solves.

Do not edit those by hand. Regenerate them.
