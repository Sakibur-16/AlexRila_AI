"""Version constants.

Three versions evolve independently and are all reported in responses:

* ``API_VERSION``      -- URL-visible contract (``/api/v1``). Changes only on a
  breaking change to request/response *envelopes*.
* ``SCHEMA_VERSION``   -- the ``Receipt`` document contract consumed by backends.
  Minor bumps add optional fields; major bumps may remove/retype fields.
* ``PIPELINE_VERSION`` -- extraction/normalisation behaviour. Bumped whenever
  output *values* may change for identical input, even if the shape does not.

Never change output semantics without bumping ``PIPELINE_VERSION``: golden-test
diffs are reviewed against it.
"""

from __future__ import annotations

from typing import Final

API_VERSION: Final[str] = "v1"
SCHEMA_VERSION: Final[str] = "1.0"
PIPELINE_VERSION: Final[str] = "1.0.0"

#: Version of the LLM extraction prompt template. Recorded in processing
#: metadata so a regression can be traced to a prompt change.
PROMPT_VERSION: Final[str] = "1.0.0"
