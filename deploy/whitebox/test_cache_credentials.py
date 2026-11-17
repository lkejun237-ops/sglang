"""Workload token rotation contract; synthetic credentials, no network access."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from botocore.exceptions import CredentialRetrievalError

from sglang.multimodal_gen.runtime.realtime.cache_credentials import DLCFileCredentials


def test_dlc_rotation_is_reread_instead_of_freezing_submission_credentials(tmp_path):
    path = tmp_path / "credentials"
    provider = DLCFileCredentials(path)
    assert provider.load() is None
    value = dict(
        AccessKeyId="old-key",
        AccessKeySecret="old-secret",
        SecurityToken="old-token",
        Expiration=(datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
    )
    path.write_text(json.dumps(value))
    credentials = provider.load()
    assert credentials.get_frozen_credentials().access_key == "old-key"
    value.update(
        AccessKeyId="new-key",
        AccessKeySecret="new-secret",
        SecurityToken="new-token",
        Expiration=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )
    path.write_text(json.dumps(value))
    actual = credentials.get_frozen_credentials()
    assert (actual.access_key, actual.secret_key, actual.token) == (
        "new-key",
        "new-secret",
        "new-token",
    )
    path.write_text('{"AccessKeySecret":"DO-NOT-LOG"}')
    with pytest.raises(CredentialRetrievalError) as caught:
        provider.read()
    assert "DO-NOT-LOG" not in str(caught.value)
