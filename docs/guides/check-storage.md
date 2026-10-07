# Checking Storage for team2pop

Commands for checking disk usage in the `team2pop` home directory (`/home/jupyter-team2pop`).

## Overall disk / filesystem usage

```bash
df -h /home/jupyter-team2pop        # free/used space on the filesystem holding your home
df -h                               # all mounted filesystems
```

## Total size of your home directory

```bash
du -sh /home/jupyter-team2pop
```

## Biggest folders in your home (including hidden ones like .cache, .conda)

```bash
du -sh /home/jupyter-team2pop/{*,.[!.]*} 2>/dev/null | sort -rh | head -20
```

## Drill into a specific folder

```bash
du -h --max-depth=1 /home/jupyter-team2pop/Mars 2>/dev/null | sort -rh | head -20
```

## Find the largest individual files (>500MB)

```bash
find /home/jupyter-team2pop -type f -size +500M -exec ls -lh {} \; 2>/dev/null | sort -k5 -rh
```

## Per-user quota (if quotas are enabled)

```bash
quota -s
```

## Usual space hogs worth checking

```bash
du -sh ~/.cache ~/.conda ~/.local ~/.npm ~/.nvm ~/git-backups-2026-10-06 2>/dev/null
conda clean --all --dry-run          # shows what conda cleanup would free
pip cache info                       # size of the pip cache
```

## Interactive browser (if installed)

```bash
ncdu /home/jupyter-team2pop
```
