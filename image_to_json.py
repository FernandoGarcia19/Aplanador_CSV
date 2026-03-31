from openai import OpenAI
import base64

BASE_PROMPT = """
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

def create_prompt_with_image(image_base64):
    """Create Chat Completions-compatible messages with image + instructions."""
    return [
        {
            "role": "system",
            "content": BASE_PROMPT,
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
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        },
    ]


def image_to_json(image):
    """Obtain a response from the model by sending the prompt with the embedded image."""
    client = OpenAI()
    messages = create_prompt_with_image(image)
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=messages
    )
    
    return response.choices[0].message.content

