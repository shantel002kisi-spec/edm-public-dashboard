"""Publish the v17 Priority locations visual dashboard directly from Colab.

Paste this entire file into one Google Colab cell and run it. When the file
chooser opens, select the app.py saved in the computer's Downloads folder.
"""

import base64
from getpass import getpass

import requests
from google.colab import files


OWNER = "shantel002kisi-spec"
REPOSITORY = "edm-public-dashboard"
BRANCH = "main"
REPOSITORY_PATH = "app.py"

REQUIRED_MARKERS = [
    "2026-09-07-priority-visual-story-v17",
    "The priority story at a glance",
    "priority_recorded_year_overview",
    "priority_company_year_heatmap",
    "priority_relationship_sankey",
    "priority_outlet_year_timeline",
]


print("Select C:\\Users\\shant\\Downloads\\app.py")
uploaded_files = files.upload()

if not uploaded_files:
    raise ValueError("No app.py was selected.")
if len(uploaded_files) != 1:
    raise ValueError("Select only one file: the new Downloads/app.py.")

uploaded_name, app_bytes = next(iter(uploaded_files.items()))
try:
    app_text = app_bytes.decode("utf-8")
except UnicodeDecodeError as error:
    raise ValueError("The selected file is not a UTF-8 Python file.") from error

compile(app_text, "app.py", "exec")
missing_markers = [
    marker for marker in REQUIRED_MARKERS if marker not in app_text
]
if missing_markers:
    raise ValueError(
        "This is not the v17 visual dashboard. Missing markers: "
        + ", ".join(missing_markers)
    )

print(f"Validated {uploaded_name}: v17 visual dashboard")

github_token = getpass(
    "Paste a GitHub token with Contents read/write permission, then press Enter: "
)
headers = {
    "Authorization": f"Bearer {github_token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
api_url = (
    f"https://api.github.com/repos/{OWNER}/{REPOSITORY}/contents/"
    f"{REPOSITORY_PATH}"
)

try:
    current = requests.get(
        api_url,
        headers=headers,
        params={"ref": BRANCH},
        timeout=60,
    )
    if current.status_code not in (200, 404):
        raise RuntimeError(
            f"GitHub check failed ({current.status_code}): {current.text}"
        )

    payload = {
        "message": "Add user-friendly Priority locations visuals",
        "content": base64.b64encode(app_bytes).decode("ascii"),
        "branch": BRANCH,
    }
    if current.status_code == 200:
        payload["sha"] = current.json()["sha"]

    response = requests.put(
        api_url,
        headers=headers,
        json=payload,
        timeout=180,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub upload failed ({response.status_code}): {response.text}"
        )

    result = response.json()
    print("\nV17 PRIORITY VISUALS PUBLISHED TO GITHUB")
    print("Commit:", result["commit"]["sha"])
    print(
        f"https://github.com/{OWNER}/{REPOSITORY}/blob/{BRANCH}/app.py"
    )
    print("Next: wait for Streamlit to rebuild, then reboot the app if needed.")
finally:
    github_token = ""
    headers["Authorization"] = ""
