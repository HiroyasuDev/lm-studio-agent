from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 1. Connect to local LM Studio or Ollama (via OpenAI compatibility)
# Make sure your local server is turned ON in LM Studio!
local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="local-no-key",
    model="local-model", # Usually ignored by local endpoints, but required by Langchain
    temperature=0.7
)

# 2. Define your Agents (The Team)
researcher = Agent(
    role='Hardware AI Researcher',
    goal='Discover the optimal ways to run LLMs on an i7 CPU with 64GB RAM.',
    backstory='You are an expert in local AI infrastructure, CPU offloading, and hardware optimization.',
    verbose=True,
    allow_delegation=False, # We want them doing their own work for this simple setup
    llm=local_llm
)

writer = Agent(
    role='Technical Blogger',
    goal='Write an engaging blog post based on the researcher\'s findings.',
    backstory='You translate complex hardware and AI concepts into simple, engaging articles.',
    verbose=True,
    allow_delegation=False,
    llm=local_llm
)

# 3. Define their Tasks
task1 = Task(
    description='Investigate how CPU offloading combined with 64GB RAM overcomes VRAM limitations for models like Mistral Nemo.',
    expected_output='A bulleted list of 3-5 technical findings.',
    agent=researcher
)

task2 = Task(
    description='Using the researcher\'s findings, write a short 2-paragraph blog post on why RAM is the secret weapon for local AI.',
    expected_output='A 2-paragraph Markdown formatted summary.',
    agent=writer
)

# 4. Assemble the Crew and start the process
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential # Hand off logically from researcher -> writer
)

if __name__ == "__main__":
    print("Starting CrewAI Multi-Agent Workflow...")
    result = tech_crew.kickoff()
    
    print("\n\n" + "="*40)
    print("FINAL DELIVERABLE")
    print("="*40)
    print(result)
