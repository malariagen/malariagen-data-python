#!/usr/bin/env bash

set -xeuo pipefail
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-469.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-469.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --path-update=false --quiet --no-compile-python
