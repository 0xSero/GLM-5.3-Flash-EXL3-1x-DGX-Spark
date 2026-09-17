#!/usr/bin/env bash
# download.sh — fetch the weights into ./models/ on this machine only.
# Same as ./start.sh download. Re-runs resume; complete weights skip.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
