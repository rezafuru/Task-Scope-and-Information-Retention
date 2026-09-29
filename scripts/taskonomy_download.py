#!/usr/bin/env python3
r"""Run the official Omnidata downloader with URL-safe license form fields.

The downloader interpolates the license acceptance answers into a form URL without escaping
them, so answers containing spaces or punctuation produce a malformed request. This wraps the
template string to percent-encode every field before interpolation and then calls the
downloader unchanged. Every other option is the Omnidata downloader's own, so pass its flags
through, for example:

  python scripts/taskonomy_download.py rgb depth_zbuffer segment_semantic mask_valid \
      --components taskonomy --subset tiny --split all \
      --dest /path/to/taskonomy_tiny --dest_compressed /path/to/taskonomy_tiny_archives \
      --name '<licensed user>' --email '<licensed email>'

Install the downloader first with `pip install omnidata-tools`.
"""

from __future__ import annotations

import importlib
import sys
from urllib.parse import quote


class _EncodedURLTemplate(str):
    def format(self, *args: object, **kwargs: object) -> str:
        encoded = {key: quote(str(value), safe="") for key, value in kwargs.items()}
        return super().format(*args, **encoded)


def main() -> None:
    help_requested = bool({"-h", "--help"} & set(sys.argv[1:]))
    try:
        download_module = importlib.import_module("omnidata_tools.dataset.download")
    except ImportError as error:
        if help_requested:
            print(__doc__)
            return
        raise SystemExit(
            "The Taskonomy download needs the omnidata-tools package "
            "(pip install omnidata-tools)."
        ) from error
    download_module.GOOGLE_FORM_URL = _EncodedURLTemplate(download_module.GOOGLE_FORM_URL)
    download_module.download()


if __name__ == "__main__":
    main()
