
"""Offline comparison of ABSA prompt strategies. Makes no API calls."""

from .prompt_strategies import load_prompt, PROMPT_FILES


def main():
    print("\nPROMPT STRATEGY COMPARISON")
    print("=" * 55)

    for strategy, filename in PROMPT_FILES.items():
        try:
            prompt = load_prompt(strategy)
            print(f"\n{strategy.upper()}")
            print(f"File: prompts/{filename}")
            print(f"Characters: {len(prompt)}")
            print("Status: READY")
            print("-" * 55)
            print(prompt[:250].replace("\n", " "))
            print("...")
        except ValueError as exc:
            print(f"\n{strategy.upper()}")
            print(f"File: prompts/{filename}")
            print(f"Status: BLOCKED")
            print(f"Reason: {exc}")

    print("\nOffline comparison complete. No API calls made.")


if __name__ == "__main__":
    main()
