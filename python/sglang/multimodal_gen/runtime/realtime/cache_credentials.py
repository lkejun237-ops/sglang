# SPDX-License-Identifier: Apache-2.0
"""Use PAI's rotating workload credentials without copying a personal AccessKey."""

import json
from pathlib import Path

from botocore.credentials import CredentialProvider, RefreshableCredentials
from botocore.exceptions import CredentialRetrievalError


class DLCFileCredentials(CredentialProvider):
    METHOD = "aliyun-dlc-file"

    def __init__(self, path=Path("/mnt/.alibabacloud/credentials")):
        self.path = Path(path)

    def read(self):
        try:
            if self.path.stat().st_size > 65536:
                raise ValueError()
            value = json.loads(self.path.read_text())
            fields = {
                "access_key": "AccessKeyId",
                "secret_key": "AccessKeySecret",
                "token": "SecurityToken",
                "expiry_time": "Expiration",
            }
            result = {key: value[name] for key, name in fields.items()}
            if any(not isinstance(v, str) or not v for v in result.values()):
                raise ValueError()
            return result
        except (OSError, KeyError, ValueError, TypeError):
            raise CredentialRetrievalError(
                provider=self.METHOD,
                error_msg="The PAI workload credential file is unavailable or invalid",
            ) from None

    def load(self):
        if not self.path.exists():
            return None
        return RefreshableCredentials.create_from_metadata(
            metadata=self.read(),
            refresh_using=self.read,
            method=self.METHOD,
        )


def storage_session():
    import boto3
    import botocore.session

    session = botocore.session.get_session()
    # Explicit deployment environment credentials still take precedence. Other
    # providers remain unchanged when this is not a PAI workload.
    session.get_component("credential_provider").insert_after(
        "env", DLCFileCredentials()
    )
    return boto3.Session(botocore_session=session)
