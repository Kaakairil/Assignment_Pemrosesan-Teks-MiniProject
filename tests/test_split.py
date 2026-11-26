"""Test split.py: reject invalid data, no label_model in src/."""
import subprocess


def test_no_label_model_in_src():
    """grep -r 'label_model' src/ harus tidak menemukan apa pun."""
    result = subprocess.run(
        ["grep", "-r", "label_model", "src/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"label_model ditemukan di src/:\n{result.stdout}"
