import json
import re


MAX_FACT_CHECK_ATTEMPTS = 6


def clean_json_response(text):
    """
    Clean common Qwen formatting mistakes before JSON parsing.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "")

    # Remove common prefixes.
    text = re.sub(
        r"^(json|result|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def extract_json_object(text):
    """
    Extract the largest balanced JSON object from Qwen output.

    Unlike a simple regex, this understands quoted strings,
    escaped quotes and nested braces.
    """

    text = clean_json_response(text)

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in Qwen response."
        )

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):

        char = text[i]

        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    raise ValueError(
        "JSON object appears to be incomplete."
    )


def parse_json_response(text):
    """
    Parse Qwen's JSON response.
    """

    cleaned = clean_json_response(text)

    # First try the entire response.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Then extract a balanced JSON object.
    extracted = extract_json_object(cleaned)

    try:
        return json.loads(extracted)

    except json.JSONDecodeError as e:

        raise ValueError(
            "Found a JSON-like block but failed to parse it: "
            f"{e}"
        )


def validate_fact_check(data):
    """
    Validate the fact-check structure.
    """

    if not isinstance(data, dict):
        raise ValueError(
            "Fact-check result must be a JSON object."
        )

    if "claims" not in data:
        raise ValueError(
            "Fact-check result is missing 'claims'."
        )

    if not isinstance(data["claims"], list):
        raise ValueError(
            "'claims' must be a list."
        )

    for index, claim in enumerate(
        data["claims"],
        1,
    ):

        if not isinstance(claim, dict):
            raise ValueError(
                f"Claim {index} is not an object."
            )

        required = [
            "claim",
            "classification",
            "evidence",
        ]

        for field in required:

            if field not in claim:
                raise ValueError(
                    f"Claim {index} is missing "
                    f"'{field}'."
                )

        classification = str(
            claim["classification"]
        ).lower().strip()

        allowed = {
            "verified",
            "supported",
            "uncertain",
            "unsupported",
            "false",
            "misleading",
        }

        if classification not in allowed:
            raise ValueError(
                f"Claim {index} has invalid "
                f"classification: {classification}"
            )

    return True


def build_prompt(
    topic,
    research,
    evidence_dossier,
):
    return f"""
You are a historical fact-checking assistant.

Your task is to evaluate the evidence dossier against
the supplied research material.

TOPIC:
{topic}

RESEARCH MATERIAL:
{research}

EVIDENCE DOSSIER:
{evidence_dossier}

For every important factual claim:

1. Identify the claim.
2. Determine whether the supplied evidence supports it.
3. Classify it as one of:

   verified
   supported
   uncertain
   unsupported
   false
   misleading

4. Explain the evidence briefly.
5. Identify the source supporting the claim when possible.

IMPORTANT:

- Do not invent sources.
- Do not invent evidence.
- Do not introduce unrelated historical claims.
- If evidence is insufficient, use "uncertain".
- Distinguish traditional/cultural claims from independently
  verified historical facts.
- Do not treat a traditional account as automatically false.
- Do not treat a traditional account as independently verified
  historical fact unless the evidence supports that conclusion.

RETURN ONLY VALID JSON.

Do not use Markdown.
Do not use ```json.
Do not write anything before or after the JSON.

The response MUST have exactly this general structure:

{{
  "overall_status": "PASS or REVIEW",
  "claims": [
    {{
      "claim": "factual claim",
      "classification": "verified",
      "evidence": "brief evidence explanation",
      "source": "source or source identifier",
      "notes": "optional note"
    }}
  ],
  "summary": "overall fact-check summary"
}}

CRITICAL JSON RULES:

- Use double quotes for all JSON keys and string values.
- Escape quotation marks inside strings.
- Do not place trailing commas.
- Every object must have matching braces.
- Every array must have matching brackets.
- Return syntactically valid JSON.
"""


def run(
    topic,
    research,
    evidence_dossier=None,
    config=None,
    qwen=None,
):
    """
    Fact-check the evidence dossier.

    Qwen performs the reasoning.
    Python validates the returned structure.
    """

    if qwen is None:
        raise ValueError(
            "Qwen client is required for fact checking."
        )

    # Backward compatibility:
    # Some pipeline versions may pass the dossier under
    # a different variable or omit it.
    if evidence_dossier is None:
        evidence_dossier = research

    prompt = build_prompt(
        topic=topic,
        research=research,
        evidence_dossier=evidence_dossier,
    )

    last_error = None

    for attempt in range(
        1,
        MAX_FACT_CHECK_ATTEMPTS + 1,
    ):

        print(
            f"[FACT_CHECK] ATTEMPT "
            f"{attempt}/{MAX_FACT_CHECK_ATTEMPTS}"
        )

        current_prompt = prompt

        # ----------------------------------------------------
        # After a failure, explicitly tell Qwen what went wrong.
        # ----------------------------------------------------

        if last_error is not None:

            current_prompt += f"""

PREVIOUS ATTEMPT FAILED.

Parser error:
{last_error}

Generate the JSON again from scratch.

Do not repair the previous response manually.
Return ONLY a complete valid JSON object.
"""

        result = qwen.generate(
            current_prompt,
            max_new_tokens=2200,
            temperature=0.15,
        )

        try:

            data = parse_json_response(result)

            validate_fact_check(data)

            print(
                f"[FACT_CHECK] ACCEPTED | "
                f"{len(data['claims'])} claims"
            )

            return data

        except Exception as e:

            last_error = str(e)

            print(
                f"[FACT_CHECK] ATTEMPT {attempt} "
                f"FAILED | {last_error}"
            )

    raise ValueError(
        f"Could not extract valid JSON after "
        f"{MAX_FACT_CHECK_ATTEMPTS} attempts: "
        f"{last_error}"
    )
