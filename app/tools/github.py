import json
import re
import urllib.request


def check_github_repo(repo: str = "kishorejorige/pocketagent") -> str:
    """List the top-level files in a public GitHub repository and warn if .env is public.

    Args:
        repo: Repository as owner/name, for example kishorejorige/pocketagent.
    """
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo):
        return "Invalid repository format. Use owner/name."
    url = f"https://api.github.com/repos/{repo}/contents/"
    req = urllib.request.Request(url, headers={"User-Agent": "PocketAgent"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            items = json.load(r)
    except Exception as e:
        return f"Could not read repo: {e}"
    names = sorted(i["name"] for i in items)
    warning = " WARNING: .env is public!" if ".env" in names else ""
    return "Files: " + ", ".join(names) + warning