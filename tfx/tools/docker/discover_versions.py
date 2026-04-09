import re
import subprocess
import json
import sys

def discover():
    # In the discovery container, we expect dependencies.py to be in /src
    sys.path.append('/src')

    try:
        import dependencies
    except ImportError:
        # Fallback for different layouts
        sys.path.append('/src/tfx')
        import dependencies

    packages = dependencies.make_required_install_packages()

    # Find TensorFlow requirement
    tf_req = [p for p in packages if p.startswith("tensorflow") and not p.startswith("tensorflow-")][0]
    # Find Beam requirement
    beam_req = [p for p in packages if "apache-beam" in p][0]

    # Detect TF version for base image selection
    tf_version_match = re.findall(r"([0-9]+\.[0-9]+\.[0-9]+|[0-9]+\.[0-9]+)", tf_req)
    tf_version = tf_version_match[0] if tf_version_match else "unknown"

    # Detect Beam version using pip dry-run report for reliability
    beam_version = "unknown"
    try:
        # Use --report which returns JSON, much more reliable than parsing text
        cmd = ["python3", "-m", "pip", "install", beam_req, "--dry-run", "--report", "-", "-c", "/src/tfx/tools/docker/requirements.txt", "-c", "/src/tfx/tools/docker/build_constraints.txt"]
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode()
        report = json.loads(output)
        for pkg in report.get("install", []):
            if pkg.get("metadata", {}).get("name") == "apache-beam":
                beam_version = pkg.get("metadata", {}).get("version")
                break
    except Exception:
        # Fallback to text parsing
        try:
            cmd = ["python3", "-m", "pip", "install", beam_req, "--dry-run", "-c", "/src/tfx/tools/docker/requirements.txt", "-c", "/src/tfx/tools/docker/build_constraints.txt"]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
            match = re.search(r"apache-beam-([0-9]+\.[0-9]+\.[0-9]+)", output)
            if match:
                beam_version = match.group(1)
        except Exception:
            pass

    # Final fallback to regex on the requirement string itself if all else fails
    if beam_version == "unknown":
        beam_matches = re.findall(r"([0-9]+\.[0-9]+\.[0-9]+|[0-9]+\.[0-9]+)", beam_req)
        beam_version = beam_matches[0] if beam_matches else "2.53.0"

    print(f"{tf_version}|{beam_version}")

if __name__ == "__main__":
    discover()
