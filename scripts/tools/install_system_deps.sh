#!/usr/bin/env bash
set -euo pipefail
if command -v apt-get >/dev/null 2>&1; then
apt-get update
apt-get install -y aria2 curl ffmpeg git-lfs jq libgl1 libglib2.0-0 libsndfile1 mediainfo p7zip-full sox tar unzip wget
elif command -v yum >/dev/null 2>&1; then
yum install -y aria2 curl ffmpeg git-lfs jq libgl1 libglib2.0-0 libsndfile1 mediainfo p7zip-full sox tar unzip wget
else
echo "unsupported package manager"; exit 1
fi
