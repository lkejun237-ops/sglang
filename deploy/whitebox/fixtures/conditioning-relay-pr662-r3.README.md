# Conditioning collector test baseline

`conditioning-relay-pr662-r3.py.txt` is the verbatim, dependency-free `relay.py`
from the T051 PR #662 r3 collector archive, used only as an immutable test input.
It is not a second editable implementation or a deployed module.

- Source archive SHA256: `a80254563127d1bc76448a390b96ae4984ca7b7100e68371c88cf6056682f402`
- Source file SHA256: `fa97a2ea9e590e60fd00b12130e053b079f0265862a67200bf2f83ee7c73831b`
- Source preparation: SGLang PR #662 commit `cbf86ca70b681fcfe350e59b90f63d9f7db7ccef`.

Tests pass this exact source through the same `patched_sources` function as the
release builder; a hash mismatch fails instead of skipping. The snapshot needs
no credentials, private download, cloud SDK, model or GPU. It retains the Room
rules so regression tests can verify that conditioning changes do not widen them.

When reviewing a new collector baseline, replace the snapshot and update the
builder's pins together. Full release archive checksum and preservation checks
remain separate from these always-running source/reader tests. `.gitattributes`
preserves the pinned bytes across Windows and Linux checkouts.
