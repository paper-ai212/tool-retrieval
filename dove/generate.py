"""Generate V1 source-dialect rewrites and three V2 doc2query views. Resume-safe."""

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = os.environ["UNMIX_LLM_URL"]
GENERATION_MODEL = os.environ["UNMIX_LLM_MODEL"]
DOVE_DIR = os.environ["UNMIX_DOVE_DIR"]
ANCHORS_PATH = os.environ["UNMIX_ANCHORS"]
SYSTEM = (
    "You rewrite target tool documentation into the source-ecosystem documentation style used by a dense retriever. "
    "The retriever was trained on tool docs of this exact shape: a JSON object with keys "
    "'name', 'description', 'parameters', and 'tool_profile' = {'function': <one concise functional sentence>, 'tags': [...]}. "
    "Description/tool_profile sentences must read like natural English explanations of WHAT the tool does and WHEN to use it "
    "(not raw code/schema).\n\n"
    "Rules:\n"
    "1. NEVER invent capabilities, endpoints, parameters, or behavior not present in the source documentation.\n"
    "2. Keep the exact tool name and every original parameter/argument (copy them verbatim when present).\n"
    "3. Description: 3 natural sentences saying what the tool does and the typical user situation (like the anchors).\n"
    "4. tool_profile.function: ONE short sentence in present tense ('Returns ...', 'Creates ...', 'Identifies ...').\n"
    "5. tags: 4-8 short lowercase keywords capturing domain + core capability.\n"
    "6. Output a JSON object with exactly two keys:\n"
    "   'doc': the converted documentation object (not a string),\n"
    "   'user_queries': exactly 3 natural user requests (each 8-40 words) that this tool should be retrieved for, "
    "   phrased like real user instructions, without tool names.\n"
    "Output ONLY valid JSON."
)


def valid_rewrite(value):
    if not isinstance(value, dict) or not isinstance(value.get("doc"), dict):
        return False
    queries = value.get("user_queries")
    return (
        isinstance(queries, list)
        and len(queries) == 3
        and all(isinstance(query, str) and len(query.strip()) >= 8 for query in queries)
    )


def call_api(
    user_message, api_key, timeout=180, retries=5, sleep_base=2.0, max_tokens=2000
):
    body = json.dumps(
        {
            "model": GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
    ).encode()
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                API_URL,
                data=body,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = json.load(response)
            content = response_body["choices"][0]["message"].get("content", "")
            parsed = json.loads(content)
            rewritten_document = parsed.get("doc")
            generated_queries = parsed.get("user_queries")
            result = {"doc": rewritten_document, "user_queries": generated_queries}
            if not valid_rewrite(result):
                raise ValueError("expected a document and exactly 3 valid queries")
            result["user_queries"] = [query.strip() for query in generated_queries]
            return result
        except Exception as error:
            last_error = error
            time.sleep(sleep_base * (attempt + 1) * (1 + (attempt % 3)))
    raise RuntimeError("rewrite failed: %s" % last_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", default=f"{DOVE_DIR}/rewrites.json")
    parser.add_argument("--set", default=f"{DOVE_DIR}/full_set.jsonl")
    args = parser.parse_args()
    api_key = os.environ["UNMIX_LLM_API_KEY"].strip()
    rows = [json.loads(line) for line in open(args.set)]
    anchors = json.load(open(ANCHORS_PATH))
    rewrites = {}
    if os.path.exists(args.out):
        rewrites = json.load(open(args.out))
    pending_tools = [
        tool
        for tool in rows
        if not valid_rewrite(rewrites.get(tool["id"]))
    ]
    print(
        "total=%d todo=%d done=%d"
        % (len(rows), len(pending_tools), len(rows) - len(pending_tools)),
        flush=True,
    )
    # Include source-ecosystem style anchors and the target document in each request.
    anchor_text = "\n\n".join(
        "### Source-ecosystem style anchor ('query' shows the kind of user query the document must match):\n"
        f"DOC:\n{anchor['response']}\nUSER QUERY THAT THIS DOC ANSWERS:\n{anchor['query']}"
        for anchor in anchors
    )
    start_time = time.time()
    completed_count = 0
    failures = []

    def generate_one_tool(tool):
        try:
            return tool["id"], call_api(
                f"{anchor_text}\n\n### Tool documentation to convert:\n{tool['doc']}",
                api_key,
            )
        except Exception as error:
            return tool["id"], ("__ERR__", str(error))

    # Generate views concurrently and save progress as requests complete.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_one_tool, tool) for tool in pending_tools]
        for future in as_completed(futures):
            tool_id, result = future.result()
            if isinstance(result, tuple) and result and result[0] == "__ERR__":
                rewrites[tool_id] = "__FAIL__"
                failures.append((tool_id, result[1][:120]))
            else:
                rewrites[tool_id] = result
            completed_count += 1
            if completed_count % 50 == 0:
                json.dump(rewrites, open(args.out, "w"), ensure_ascii=False, indent=0)
                print(
                    "  %d/%d done %.0fs fails=%d"
                    % (
                        completed_count,
                        len(pending_tools),
                        time.time() - start_time,
                        len(failures),
                    ),
                    flush=True,
                )
    json.dump(rewrites, open(args.out, "w"), ensure_ascii=False, indent=0)
    successful_count = sum(
        1
        for tool in rows
        if valid_rewrite(rewrites.get(tool["id"]))
    )
    print(
        "DONE %.0fs ok=%d/%d fails=%d"
        % (time.time() - start_time, successful_count, len(rows), len(failures))
    )
    if failures:
        print("sample fails:", failures[:3])


if __name__ == "__main__":
    main()
