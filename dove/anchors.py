"""Select source-ecosystem style anchors for DOVE."""

import json
import os
import random

SOURCE_PAIRS_PATH = os.environ["UNMIX_SOURCE_PAIRS"]
ANCHORS_PATH = os.environ["UNMIX_ANCHORS"]


def pick_anchors(count=2):
    # Select source examples with the expected documentation fields.
    training_pairs = json.load(open(SOURCE_PAIRS_PATH))
    candidates = []
    for training_pair in training_pairs:
        try:
            document = json.loads(training_pair["response"])
        except Exception:
            continue
        if not isinstance(document, dict):
            continue
        profile = document.get("tool_profile") or {}
        function_description = (
            profile.get("function") if isinstance(profile, dict) else None
        )
        tags = profile.get("tags") if isinstance(profile, dict) else None
        description = document.get("description")
        if not (
            isinstance(function_description, str)
            and len(function_description) > 20
            and isinstance(tags, list)
            and len(tags) >= 3
            and isinstance(description, str)
            and 60 < len(description) < 500
        ):
            continue
        if len(training_pair["response"]) > 1600:
            continue
        candidates.append(training_pair)
    random.Random(0).shuffle(candidates)
    return [
        {"query": candidate["query"], "response": candidate["response"]}
        for candidate in candidates[:count]
    ]


def main():
    anchors = pick_anchors(2)
    json.dump(anchors, open(ANCHORS_PATH, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
