import re
import subprocess


def main():
    output = subprocess.run(
        # ["fuser", "-v", "/dev/nvidia2"],
        "fuser -v /dev/nvidia3",
        shell=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        check=True
    )
    for line in output.stdout.decode("utf8").splitlines():
        m = re.search(r"\d+", line)
        if m is not None:
            pid = int(m.group(0))
            print(pid)
            subprocess.call(f"kill {pid}", shell=True, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    main()