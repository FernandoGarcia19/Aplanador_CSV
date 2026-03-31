from openai import OpenAI

BASE_PROMPT = """
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
"""

def create_prompt_with_image_and_json(image_base64, json_schema):
    """Create Chat Completions-compatible messages with image + schema + instructions."""
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
                    "text": f"JSON hierarchical schema:\n{json_schema}",
                },
                {
                    "type": "text",
                    "text": "Use the schema and image together. Return ONLY the CSV sample.",
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

def json_to_csv(image, json_schema):
    """Obtain a response from the model by sending the prompt with the embedded image and JSON schema."""
    client = OpenAI()
    messages = create_prompt_with_image_and_json(image, json_schema)
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=messages
    )
    
    return response.choices[0].message.content.strip()