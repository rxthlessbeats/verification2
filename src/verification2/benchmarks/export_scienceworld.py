"""Write one row per ScienceWorld variation. Dev variations stay with train."""

import json
import os
import sys
from pathlib import Path


def main(target):
    from jdk4py import JAVA_HOME

    os.environ["JAVA_HOME"] = str(JAVA_HOME)
    os.environ["PATH"] = str(JAVA_HOME) + "/bin:" + os.environ.get("PATH", "")
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()
    rows = []
    try:
        for name in env.get_task_names():
            env.load(name, 0, simplificationStr="easy", generateGoldPath=False)
            groups = {
                "train": env.get_variations_train(),
                "dev": env.get_variations_dev(),
                "test": env.get_variations_test(),
            }
            for native, variations in groups.items():
                partition = "test" if native == "test" else "train"
                for variation in variations:
                    rows.append(
                        {
                            "task_id": str(variation),
                            "domain": name,
                            "partition": partition,
                            "native_partition": native,
                            "simplification": "easy",
                        }
                    )
    finally:
        closer = getattr(env, "close", None)
        if closer:
            closer()
    Path(target).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"exported {len(rows)} scienceworld tasks", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
