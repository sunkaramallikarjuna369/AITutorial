"""
Module 17: AI Agents
====================
Autonomous AI systems that can reason, plan, and take actions!
"""

def explain_agents():
    print("=" * 60)
    print("AI Agents - Autonomous AI Systems")
    print("=" * 60)
    print("""
AI Agents are AI systems that can ACT autonomously!

Regular AI (ChatGPT): Answers questions
AI Agent: Actually DOES things for you!

Key Components:
1. Brain (LLM): Makes decisions
2. Tools: Search, code execution, APIs
3. Memory: Remembers past actions
4. Planning: Creates step-by-step plans

Agent Types:
- ReAct: Reason + Act (think, then do)
- Tool-Using: Uses external tools
- Planning: Creates and executes plans
- Multi-Agent: Multiple agents working together

Popular Frameworks:
- LangChain
- AutoGPT
- CrewAI
- LangGraph
- AutoGen
    """)

code_examples = '''
"""
AI Agent Examples
=================
"""

# ============================================
# 1. Simple ReAct Agent
# ============================================

class ReActAgent:
    """Reason + Act Agent"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def run(self, query, max_steps=5):
        observation = query
        
        for step in range(max_steps):
            # Think
            thought = self.think(observation)
            print(f"Thought: {thought}")
            
            if "FINAL ANSWER" in thought:
                return thought
            
            # Act
            action, action_input = self.parse_action(thought)
            observation = self.tools[action](action_input)
            print(f"Observation: {observation}")
            
            self.memory.append({
                'thought': thought,
                'action': action,
                'observation': observation
            })
        
        return "Max steps reached"


# ============================================
# 2. LangChain Agent
# ============================================

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

def create_langchain_agent():
    llm = OpenAI(temperature=0)
    
    tools = [
        Tool(
            name="Search",
            func=DuckDuckGoSearchRun().run,
            description="Search the internet"
        ),
        Tool(
            name="Calculator",
            func=lambda x: str(eval(x)),
            description="Do math calculations"
        )
    ]
    
    agent = initialize_agent(
        tools, llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    
    return agent


# ============================================
# 3. Multi-Agent System with AutoGen
# ============================================

import autogen

def create_multi_agent_team():
    config = [{"model": "gpt-4", "api_key": "your-key"}]
    
    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message="You research and find information.",
        llm_config={"config_list": config}
    )
    
    writer = autogen.AssistantAgent(
        name="Writer",
        system_message="You write clear content.",
        llm_config={"config_list": config}
    )
    
    critic = autogen.AssistantAgent(
        name="Critic",
        system_message="You review and improve work.",
        llm_config={"config_list": config}
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER"
    )
    
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, writer, critic],
        messages=[],
        max_round=10
    )
    
    return groupchat


# ============================================
# 4. CrewAI Team
# ============================================

from crewai import Agent, Task, Crew

def create_crew():
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information",
        backstory="Expert at finding facts"
    )
    
    writer = Agent(
        role="Writer",
        goal="Create engaging content",
        backstory="Skilled content creator"
    )
    
    research_task = Task(
        description="Research the topic",
        agent=researcher
    )
    
    write_task = Task(
        description="Write an article",
        agent=writer
    )
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task]
    )
    
    return crew


# ============================================
# 5. Tool-Using Agent
# ============================================

class ToolAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            'search': self.search,
            'calculate': self.calculate,
            'get_weather': self.get_weather
        }
    
    def search(self, query):
        # Implement search
        return f"Search results for: {query}"
    
    def calculate(self, expression):
        return str(eval(expression))
    
    def get_weather(self, location):
        return f"Weather in {location}: Sunny, 72°F"
    
    def run(self, user_query):
        # Decide which tool to use
        tool_choice = self.llm.decide_tool(user_query, list(self.tools.keys()))
        
        # Execute tool
        result = self.tools[tool_choice['tool']](tool_choice['input'])
        
        # Generate response
        return self.llm.generate_response(user_query, result)


# ============================================
# 6. Planning Agent
# ============================================

class PlanningAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def create_plan(self, goal):
        prompt = f"""
        Goal: {goal}
        
        Create a step-by-step plan to achieve this goal.
        Format each step as: Step N: [action]
        """
        
        plan = self.llm.generate(prompt)
        steps = self.parse_plan(plan)
        return steps
    
    def execute_plan(self, goal):
        plan = self.create_plan(goal)
        results = []
        
        for step in plan:
            print(f"Executing: {step}")
            result = self.execute_step(step)
            results.append(result)
            
            # Check if we need to replan
            if not result['success']:
                plan = self.replan(goal, results)
        
        return results


# ============================================
# 7. Memory-Enhanced Agent
# ============================================

class MemoryAgent:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.memory = vector_store
        self.short_term = []
    
    def remember(self, interaction):
        self.short_term.append(interaction)
        self.memory.add(interaction)
    
    def recall(self, query, k=5):
        return self.memory.search(query, k=k)
    
    def chat(self, message):
        # Recall relevant memories
        relevant = self.recall(message)
        
        # Include in context
        context = "\\n".join([m['text'] for m in relevant])
        
        response = self.llm.generate(
            f"Context: {context}\\n\\nUser: {message}"
        )
        
        # Remember this interaction
        self.remember({
            'user': message,
            'assistant': response
        })
        
        return response


print("AI Agent examples loaded!")
print("Install: pip install langchain autogen crewai")
'''

def main():
    explain_agents()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
