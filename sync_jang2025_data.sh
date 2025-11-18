#!/bin/bash

# Sync script for Jang2025 SingleBrain data
# Syncs every 1 hour from gauss-vpn to local directory

REMOTE_PATH="gauss-vpn:/share/vault/Users/gz2294/BICAN/Data/source/Jang2025_SingleBrain/full_finetune.dim8.chk20_chunk_results/"
LOCAL_PATH="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/source/Jang2025_SingleBrain/full_finetune.dim8.chk20_chunk_results/"
SYNC_INTERVAL=3600  # 1 hour in seconds

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

echo "Starting continuous sync from $REMOTE_PATH to $LOCAL_PATH"
echo "Sync interval: 1 hour"
echo "Press Ctrl+C to stop"
echo "----------------------------------------"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Starting rsync..."

    # Run rsync with common options:
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose
    # -h: human-readable sizes
    # --progress: show progress
    # --delete: delete files in destination that don't exist in source
    rsync -avhzL --progress "$REMOTE_PATH" "$LOCAL_PATH"

    RSYNC_EXIT=$?
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    if [ $RSYNC_EXIT -eq 0 ]; then
        echo "[$TIMESTAMP] Rsync completed successfully"
    else
        echo "[$TIMESTAMP] Rsync failed with exit code $RSYNC_EXIT"
    fi

    echo "[$TIMESTAMP] Sleeping for 1 hour..."
    echo "----------------------------------------"
    sleep $SYNC_INTERVAL
done
