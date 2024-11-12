#!/usr/bin/env bash

# Array of branch names to be deleted
branches=(
  "remotes/origin/v0.01"
  "remotes/origin/v0.02"
  "remotes/origin/v0.03"
)

# Iterate over each branch and delete it both remotely and locally
for branch in "${branches[@]}"; do
  # Extract the branch name without the 'remotes/origin/' prefix
  local_branch="${branch#remotes/origin/}"

  # Delete the remote branch
  git push origin --delete "$local_branch"
  echo "Deleted remote branch: $local_branch"

  # Delete the local branch if it exists
  if git show-ref --verify --quiet "refs/heads/$local_branch"; then
    git branch -D "$local_branch"
    echo "Deleted local branch: $local_branch"
  else
    echo "Local branch $local_branch does not exist, skipping."
  fi
done

echo "All specified branches have been deleted both locally and remotely."

