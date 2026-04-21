#!/usr/bin/env python3
"""
Verify tool calling works against the local vLLM endpoint.
Run this first. If it fails, check your vLLM deployment flags:
  --enable-auto-tool-choice --tool-call-parser hermes       (for Qwen3-32B)
  --enable-auto-tool-choice --tool-call-parser llama3_json  (for Llama-3.3-70B)
"""
import sys
import json
import openai

BASE_URL = "http://localhost:8000/v1"
API_KEY = "none"


def get_first_model(client: openai.OpenAI) -> str:
    models = client.models.list()
    if not models.data:
        print("ERROR: No models found. Is vLLM running?")
        sys.exit(1)
    model_id = models.data[0].id
    print(f"  Found model: {model_id}")
    return model_id


def test_basic_inference(client: openai.OpenAI, model: str) -> bool:
    print("\n[1/3] Basic inference...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly: INFERENCE_OK"}],
        max_tokens=20,
    )
    text = resp.choices[0].message.content or ""
    ok = "INFERENCE_OK" in text
    print(f"  {'✓' if ok else '✗'} Response: {text.strip()}")
    return ok


def test_tool_call(client: openai.OpenAI, model: str) -> bool:
    print("\n[2/3] Tool call (function calling)...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=200,
    )
    choice = resp.choices[0]
    has_tool_call = (
        choice.finish_reason == "tool_calls"
        and choice.message.tool_calls is not None
        and len(choice.message.tool_calls) > 0
    )
    if has_tool_call:
        tc = choice.message.tool_calls[0]
        args = json.loads(tc.function.arguments)
        print(f"  ✓ Tool called: {tc.function.name}({args})")
    else:
        content = choice.message.content or ""
        print(f"  ✗ No tool call. finish_reason={choice.finish_reason!r}")
        print(f"    Content: {content[:200]}")
    return has_tool_call


def test_structured_output(client: openai.OpenAI, model: str) -> bool:
    print("\n[3/3] Structured output (JSON mode)...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": 'Return a JSON object with keys "status" (string) and "count" (integer). Set status to "ok" and count to 42.',
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
    )
    text = resp.choices[0].message.content or ""
    try:
        data = json.loads(text)
        ok = data.get("status") == "ok" and data.get("count") == 42
        print(f"  {'✓' if ok else '✗'} Parsed JSON: {data}")
        return ok
    except json.JSONDecodeError:
        print(f"  ✗ Not valid JSON: {text[:200]}")
        return False


def main():
    print("=" * 60)
    print("tt-agents: Tool Calling Verification")
    print("=" * 60)
    print(f"\nEndpoint: {BASE_URL}")

    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

    try:
        model = get_first_model(client)
    except Exception as e:
        print(f"ERROR connecting to {BASE_URL}: {e}")
        print("\nMake sure vLLM is running:")
        print("  cd ~/code/tt-inference-server")
        print("  python3 run.py --model Qwen3-32B --tt-device p300x2 \\")
        print("    --workflow server --docker-server --no-auth \\")
        print("    --vllm-override-args '{\"enable_auto_tool_choice\": true, \"tool_call_parser\": \"hermes\"}'")
        sys.exit(1)

    results = {
        "inference": test_basic_inference(client, model),
        "tool_call": test_tool_call(client, model),
        "structured": test_structured_output(client, model),
    }

    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    if all_pass:
        print("\n✅ All checks passed. Ready to run agent demos!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n❌ Failed: {failed}")
        if "tool_call" in failed:
            print("\nTool call failed. Check your vLLM flags:")
            print("  For Qwen3-32B:     --tool-call-parser hermes")
            print("  For Llama-3.3-70B: --tool-call-parser llama3_json")
            print("  Both require:      --enable-auto-tool-choice")
        sys.exit(1)


if __name__ == "__main__":
    main()
