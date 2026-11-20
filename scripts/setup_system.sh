#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v apt-get >/dev/null; then
  echo 'Install Docker Engine for your OS, then run make setup.' >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y docker.io make
if apt-cache show docker-compose-v2 >/dev/null 2>&1; then
  sudo apt-get install -y docker-compose-v2
elif apt-cache show docker-compose-plugin >/dev/null 2>&1; then
  sudo apt-get install -y docker-compose-plugin
else
  echo 'Install the Docker Compose v2 plugin for this OS, then run scripts/experiment.sh NAME doctor.' >&2
  exit 1
fi
sudo systemctl enable --now docker
echo 'Docker must be usable by your execution account. Verify with docker info, then scripts/experiment.sh NAME doctor.'
