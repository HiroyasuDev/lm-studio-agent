import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# 1. Connect to your local LM Studio or Ollama instance
# LM Studio default port is 1234
# Ollama default port is 11434 (use 'http://localhost:11434/v1' for OpenAI compatibility)
local_model = OpenAIModel(
    'mistral-nemo-12b-instruct', # Label for your model
    base_url='http://localhost:1234/v1', # Ensure LM Studio Server is running
    api_key='local-no-key-required'
)

# 2. Define the EXACT structured output you want
class ResearchResult(BaseModel):
    summary: str
    key_points: list[str]
    confidence_score: float

# 3. Create the Pydantic AI Agent
research_agent = Agent(
    model=local_model,
    result_type=ResearchResult,
    system_prompt=(
        "You are a meticulous IT researcher. Your job is to extract findings "
        "and return them strictly formatted to the requested schema. Ensure "
        "the confidence score is between 0.0 and 1.0."
    )
)

async def main():
    print("Sending task to local model via PydanticAI...")
    # 4. Execute the agent task
    prompt = "Explain how CPU offloading and 64GB DDR5 RAM make up for low VRAM when running LLMs."
    
    # Run the agent (this forces the local model to output valid JSON matching your BaseModel)
    result = await research_agent.run(prompt)
    
    print("\n--- Agent Result (Parsed as Python Object) ---")
    print(f"Summary: {result.data.summary}")
    print("Key Points:")
    for pt in result.data.key_points:
        print(f" - {pt}")
    print(f"Confidence Score: {result.data.confidence_score}")

if __name__ == "__main__":
    asyncio.run(main())
