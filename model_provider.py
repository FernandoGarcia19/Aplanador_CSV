from __future__ import annotations

import base64
import os
import re
from abc import ABC, abstractmethod

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from dotenv import load_dotenv


JSON_SCHEMA_PROMPT = """
You are analyzing a table extracted from a report (financial, statistical, operational, etc.).
Your task is to infer the categorical dimensional structure required to convert the table into a flattened long dataset.
The final dataset will always contain:
fecha
valor
Your job is ONLY to detect the categorical dimensions that define each numeric value.

OBJECTIVE
Identify the categorical structure of the table and represent it as generic hierarchy levels:
nv1, nv2, nv3 ... nvN
These levels come from two sources:
1 Row dimensions (hierarchical)
2 Column dimensions (usually a single level)

STEP 1 — Identify Numeric Cells
Locate the cells containing numeric values.
These represent the FACT ("valor").
Numeric cells must NOT appear in the dimension member lists.

STEP 2 — Identify Row Dimension (Hierarchical)
The left side of the table usually contains the main categorical hierarchy.
This dimension often represents things like:
accounts
product categories
economic sectors
geographic breakdowns
statistical classifications
This row dimension may contain multiple hierarchical levels.

Rules:
• Use indentation, alignment, and grouping to detect hierarchy.
• Parent rows represent higher nv levels.
• Child rows represent deeper nv levels.

STEP 3 — Identify Column Dimension
Column headers often represent entities measured across the same categories, such as:
institutions
companies
regions
years
scenarios

STEP 4 — Determine Hierarchy Depth
Combine the row hierarchy levels and the column dimension to form the final structure.
The deepest level corresponds to the column dimension.

STEP 5 — Extract Members
For each level (nvX), return the list of possible members found in the table.
Rules:
• Members must be unique
• Maximum 10 elements per level
• Ignore empty cells
• Ignore numeric values
• Ignore formatting artifacts

OUTPUT FORMAT
Return ONLY a JSON object.
Keys must be sequential:
nv1
nv2
nv3
...
nvN
"""


CSV_GENERATION_PROMPT = """
Task:  Map the provided JSON hierarchical schema to the table image to produce a flattened, sparse CSV SAMPLE.
Structural Rules:
Mandatory Columns: The output MUST end with the columns fecha and valor.
Global Attributes: Extract the date from the table header/title and populate the fecha column if there is no date inside the table data.
Fact Extraction: The valor column must contain the numerical data. Represent empty/dash cells as - and ensure negative numbers (often in red or parentheses) retain their negative sign.
Output Requirements:
Generate a CSV code block.
Header format: nv1,nv2,nv3,...,nvN,fecha,valor
Ensure the hierarchy follows the visual "indentation" or "bolding" cues from the image and the structure identified in the JSON. ONLY return the CSV sample.
Return up to only 30 rows of sample data.
Make sure to maintain the numeric value in "valor". Do not truncate it. 
In the extremely rare case that the image contains no numeric attribute for "Valor", populate the "valor" column with a placeholder value of 1.
"""


DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)


def _guess_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _normalize_image_input(image_input: bytes | str) -> tuple[bytes, str]:
    """Normalize bytes/base64/data-url input into raw bytes and mime type."""
    if isinstance(image_input, bytes):
        image_bytes = image_input
        return image_bytes, _guess_mime_type(image_bytes)

    candidate = image_input.strip()

    # data:image/png;base64,...
    data_url_match = DATA_URL_PATTERN.match(candidate)
    if data_url_match:
        mime_type = data_url_match.group("mime")
        encoded = data_url_match.group("data")
        return base64.b64decode(encoded), mime_type

    # Plain base64 string
    image_bytes = base64.b64decode(candidate)
    return image_bytes, _guess_mime_type(image_bytes)


class TableModelProvider(ABC):
    """Adapter interface for model providers used by the image pipeline."""

    @abstractmethod
    def extract_json_schema(self, image_input: bytes | str) -> str:
        """Infer the hierarchical schema from a table image."""

    @abstractmethod
    def generate_csv_sample(self, image_input: bytes | str, json_schema: str) -> str:
        """Generate flattened CSV sample from image + schema."""


class OpenAIProvider(TableModelProvider):
    """OpenAI adapter implementation."""

    def __init__(self, model_name: str | None = None) -> None:
        self._client = OpenAI()
        self._model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-5.1")

    def extract_json_schema(self, image_input: bytes | str) -> str:
        image_bytes, mime_type = _normalize_image_input(image_input)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": JSON_SCHEMA_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this table image and return ONLY the JSON object.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
        )

        return (response.choices[0].message.content or "").strip()

    def generate_csv_sample(self, image_input: bytes | str, json_schema: str) -> str:
        image_bytes, mime_type = _normalize_image_input(image_input)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": CSV_GENERATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"JSON hierarchical schema:\n{json_schema}",
                        },
                        {
                            "type": "text",
                            "text": "Return ONLY the CSV sample.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
        )

        return (response.choices[0].message.content or "").strip()


class GeminiProvider(TableModelProvider):
    """Gemini adapter implementation."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está configurada en el entorno.")

        self._client = genai.Client(api_key=api_key)
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

    def extract_json_schema(self, image_input: bytes | str) -> str:
        image_bytes, mime_type = _normalize_image_input(image_input)
        response = self._client.models.generate_content(
            model=self._model_name,
            config=genai_types.GenerateContentConfig(
                system_instruction=JSON_SCHEMA_PROMPT,
            ),
            contents=[
                genai_types.Part.from_text(
                    text="Analyze this table image and return ONLY the JSON object.",
                ),
                genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
            ],
        )

        return (response.text or "").strip()

    def generate_csv_sample(self, image_input: bytes | str, json_schema: str) -> str:
        image_bytes, mime_type = _normalize_image_input(image_input)
        response = self._client.models.generate_content(
            model=self._model_name,
            config=genai_types.GenerateContentConfig(
                system_instruction=CSV_GENERATION_PROMPT,
            ),
            contents=[
                genai_types.Part.from_text(text=f"JSON hierarchical schema:\n{json_schema}"),
                genai_types.Part.from_text(
                    text="Return ONLY the CSV sample.",
                ),
                genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
            ],
        )

        return (response.text or "").strip()


class ModelProviderFactory:
    """Factory for creating model provider adapters."""

    @staticmethod
    def create(provider_name: str) -> TableModelProvider:
        load_dotenv()
        normalized = provider_name.strip().lower()

        if normalized == "gpt":
            return OpenAIProvider()
        if normalized == "gemini":
            return GeminiProvider()

        raise ValueError(f"Proveedor de modelo no soportado: {provider_name}")
