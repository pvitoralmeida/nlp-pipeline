import emoji


NEGATIONS = {"não", "nunca", "jamais", "nem", "not", "never", "no"}


def normalize_emojis(text: str) -> str:
    return emoji.demojize(text, delimiters=(" ", " "))


def handle_negations(tokens):
    result = []
    negate = False

    for token in tokens:
        if token in NEGATIONS:
            negate = True
            result.append(token)
            continue

        if negate:
            result.append(f"{token}_NEG")
            negate = False
        else:
            result.append(token)

    return result