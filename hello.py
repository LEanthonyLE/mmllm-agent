"""A minimal entry script for git pipeline testing."""


def greet(name: str = "World") -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet())
