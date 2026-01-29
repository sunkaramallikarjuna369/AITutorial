"""
=============================================================================
MODULE 17: AI AGENTS - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of AI Agents including:
- 4W+H Explanations (What, Why, When, Where, How)
- Agent Architectures (ReAct, Tool-Use, Planning)
- Multi-Agent Systems
- Implementation Examples
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install langchain openai

For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import os
from typing import List, Dict, Callable, Optional


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class AgentAssistant:
    """Use GenAI models for agent explanations and demos"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
    
    def explain_with_openai(self, concept: str) -> str:
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY]"
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Explain {concept} in AI agents with examples."}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": f"Explain {concept} in AI agents.", "stream": False},
                timeout=60
            )
            return response.json().get("response", "") if response.status_code == 200 else "[Ollama error]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"
    
    def run_simple_agent(self, task: str) -> str:
        """Run a simple ReAct-style agent"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY to run agent]"
            
            client = OpenAI(api_key=self.openai_key)
            
            system_prompt = """You are a helpful AI agent. For each task, think step by step:

1. Thought: Analyze what needs to be done
2. Action: Decide what action to take (or "Final Answer" if done)
3. Observation: Note the result

Available actions:
- Calculate: Do math calculations
- Search: Look up information (simulated)
- Final Answer: Provide the final response

Format your response as:
Thought: [your reasoning]
Action: [action name]: [action input]
Observation: [result]
... (repeat as needed)
Final Answer: [your final response]"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task}"}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Agent Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE AI AGENTS EXPLANATION (4W+H)
# =============================================================================

def explain_agents_comprehensive():
    """Comprehensive 360-degree explanation of AI Agents"""
    print("=" * 70)
    print("AI AGENTS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT ARE AI AGENTS?
===================

AI Agents are autonomous systems that perceive their environment,
make decisions, and take actions to achieve goals.

KEY CHARACTERISTICS:
    1. Autonomy: Operate without constant human input
    2. Reactivity: Respond to environment changes
    3. Proactivity: Take initiative to achieve goals
    4. Social ability: Interact with other agents/humans

LLM-BASED AGENTS:
    LLM (brain) + Tools (capabilities) + Memory (context) = Agent
    
    The LLM reasons about what to do, tools execute actions,
    memory maintains context across interactions.

AGENT LOOP:
    Observe -> Think -> Act -> Observe -> ...

WHY ARE AGENTS IMPORTANT?
=========================

1. AUTOMATION: Handle complex, multi-step tasks
2. FLEXIBILITY: Adapt to new situations
3. SCALABILITY: Work 24/7 without fatigue
4. CAPABILITY: Access tools humans can't use as fast

WHEN TO USE AGENTS?
===================

USE AGENTS FOR:
    - Multi-step tasks requiring reasoning
    - Tasks needing external tools (search, code, APIs)
    - Open-ended problems
    - Workflows with decision points

DON'T USE AGENTS FOR:
    - Simple, single-step tasks
    - Tasks requiring perfect reliability
    - High-stakes decisions without oversight
    - When simpler solutions work

WHERE ARE AGENTS USED?
======================

APPLICATIONS:
    - Customer support (handle complex queries)
    - Research assistants (search, summarize, analyze)
    - Coding assistants (write, test, debug code)
    - Personal assistants (schedule, email, tasks)
    - Data analysis (query, visualize, report)

PRODUCTS:
    - ChatGPT with plugins/tools
    - Claude with computer use
    - Microsoft Copilot
    - AutoGPT, BabyAGI
    - Devin (coding agent)

HOW DO AGENTS WORK?
===================

BASIC ARCHITECTURE:
    1. User provides goal/task
    2. Agent plans approach
    3. Agent selects and uses tools
    4. Agent observes results
    5. Agent decides next action or completes
    
COMPONENTS:
    - LLM: Reasoning engine
    - Tools: External capabilities
    - Memory: Short-term and long-term
    - Planning: Task decomposition
    - Execution: Action taking
    """)


# =============================================================================
# SECTION 2: AGENT ARCHITECTURES
# =============================================================================

def agent_architectures():
    """Different agent architectures"""
    print("\n" + "=" * 70)
    print("AGENT ARCHITECTURES")
    print("=" * 70)
    
    print("""
1. ReAct (REASONING + ACTING)
   ===========================
   Interleave reasoning and actions
   
   Format:
   Thought: I need to find the population of France
   Action: Search("population of France")
   Observation: France has a population of 67 million
   Thought: Now I have the answer
   Action: Final Answer
   Answer: France has 67 million people
   
   Pros: Interpretable, flexible
   Cons: Can get stuck in loops

2. TOOL-USE AGENTS
   ================
   LLM decides which tool to use
   
   Tools:
   - Calculator: Math operations
   - Search: Web search
   - Code: Execute Python
   - API: Call external services
   
   Example (OpenAI Function Calling):
   functions = [
       {"name": "search", "parameters": {...}},
       {"name": "calculate", "parameters": {...}}
   ]
   
   LLM returns: {"name": "search", "arguments": {"query": "..."}}

3. PLANNING AGENTS
   ================
   Create plan before execution
   
   Plan-and-Execute:
   1. Create high-level plan
   2. Execute each step
   3. Replan if needed
   
   Example:
   Task: "Write a blog post about AI"
   Plan:
   1. Research AI trends
   2. Create outline
   3. Write introduction
   4. Write body sections
   5. Write conclusion
   6. Review and edit

4. HIERARCHICAL AGENTS
   ====================
   Manager agent delegates to worker agents
   
   Manager: Understands goal, assigns tasks
   Workers: Specialized agents for subtasks
   
   Example:
   Manager -> Research Agent -> Writing Agent -> Review Agent

5. MULTI-AGENT SYSTEMS
   ====================
   Multiple agents collaborate or compete
   
   Patterns:
   - Debate: Agents argue, reach consensus
   - Collaboration: Agents work on different parts
   - Supervision: One agent checks another's work
   
   Examples:
   - AutoGen (Microsoft)
   - CrewAI
   - LangGraph

6. REFLEXION
   ==========
   Agent reflects on failures and improves
   
   Loop:
   1. Attempt task
   2. Evaluate result
   3. Reflect on what went wrong
   4. Try again with insights
    """)


# =============================================================================
# SECTION 3: TOOLS AND CAPABILITIES
# =============================================================================

def tools_and_capabilities():
    """Agent tools and capabilities"""
    print("\n" + "=" * 70)
    print("AGENT TOOLS AND CAPABILITIES")
    print("=" * 70)
    
    print("""
COMMON TOOLS:
=============

1. SEARCH:
   - Web search (Google, Bing)
   - Document search (RAG)
   - Database queries
   
2. CODE EXECUTION:
   - Python interpreter
   - Shell commands
   - Jupyter notebooks
   
3. FILE OPERATIONS:
   - Read/write files
   - Parse documents (PDF, Word)
   - Image processing
   
4. APIS:
   - Weather, stocks, news
   - Email, calendar
   - Custom business APIs
   
5. BROWSER:
   - Navigate web pages
   - Fill forms
   - Extract information
   
6. COMMUNICATION:
   - Send emails
   - Slack messages
   - Create tickets

TOOL DEFINITION EXAMPLE:
========================

# OpenAI Function Calling format
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# LangChain Tool
from langchain.tools import Tool

search_tool = Tool(
    name="Search",
    func=search_function,
    description="Search the web for information"
)

TOOL SELECTION:
===============

LLM decides which tool based on:
- Task requirements
- Tool descriptions
- Previous observations

Best practices:
- Clear, specific tool descriptions
- Limit number of tools (cognitive load)
- Include examples in descriptions
    """)


# =============================================================================
# SECTION 4: MEMORY SYSTEMS
# =============================================================================

def memory_systems():
    """Agent memory systems"""
    print("\n" + "=" * 70)
    print("AGENT MEMORY SYSTEMS")
    print("=" * 70)
    
    print("""
TYPES OF MEMORY:
================

1. SHORT-TERM (WORKING) MEMORY:
   - Current conversation context
   - Recent observations
   - Limited by context window
   
   Implementation: Conversation history in prompt

2. LONG-TERM MEMORY:
   - Persistent across sessions
   - Facts, experiences, preferences
   - Retrieved when relevant
   
   Implementation: Vector database (RAG)

3. EPISODIC MEMORY:
   - Specific past experiences
   - "Last time I did X, Y happened"
   - Helps avoid repeating mistakes
   
   Implementation: Store and retrieve past interactions

4. SEMANTIC MEMORY:
   - General knowledge
   - Facts about the world
   - Domain expertise
   
   Implementation: Knowledge base, RAG

5. PROCEDURAL MEMORY:
   - How to do things
   - Skills and procedures
   - Tool usage patterns
   
   Implementation: Few-shot examples, fine-tuning

MEMORY MANAGEMENT:
==================

CHALLENGES:
- Context window limits
- Relevance of retrieved memories
- Memory staleness

STRATEGIES:
- Summarization: Compress old context
- Selective retrieval: Only relevant memories
- Forgetting: Remove outdated information
- Hierarchical: Summary + details

EXAMPLE IMPLEMENTATION:
=======================

class AgentMemory:
    def __init__(self):
        self.short_term = []  # Recent messages
        self.long_term = VectorStore()  # Persistent
    
    def add(self, message):
        self.short_term.append(message)
        if len(self.short_term) > 10:
            # Summarize and store
            summary = summarize(self.short_term[:5])
            self.long_term.add(summary)
            self.short_term = self.short_term[5:]
    
    def retrieve(self, query):
        # Get relevant long-term memories
        relevant = self.long_term.search(query, k=3)
        return self.short_term + relevant
    """)


# =============================================================================
# SECTION 5: MULTI-DOMAIN USE CASES
# =============================================================================

def multi_domain_use_cases():
    """Multi-domain agent use cases"""
    print("\n" + "=" * 70)
    print("MULTI-DOMAIN AGENT USE CASES")
    print("=" * 70)
    
    print("""
HEALTHCARE:
===========
- Medical research assistant
- Patient intake automation
- Clinical documentation
- Drug interaction checking

Example Agent:
    Task: "Review patient symptoms and suggest tests"
    Tools: Medical database, lab ordering system
    Memory: Patient history, guidelines

FINANCE:
========
- Financial analysis agent
- Trading assistant
- Compliance monitoring
- Customer onboarding

Example Agent:
    Task: "Analyze company financials and write report"
    Tools: SEC filings API, calculator, chart generator
    Memory: Previous analyses, market context

CUSTOMER SERVICE:
=================
- Complex query resolution
- Multi-system troubleshooting
- Escalation handling
- Proactive outreach

Example Agent:
    Task: "Help customer with billing issue"
    Tools: CRM, billing system, knowledge base
    Memory: Customer history, previous interactions

RESEARCH:
=========
- Literature review
- Data analysis
- Hypothesis generation
- Experiment design

Example Agent:
    Task: "Survey recent papers on transformer efficiency"
    Tools: arXiv search, PDF reader, note-taking
    Memory: Research context, key findings

E-COMMERCE:
===========
- Product recommendations
- Inventory management
- Pricing optimization
- Supplier negotiation

Example Agent:
    Task: "Optimize pricing for holiday season"
    Tools: Sales data, competitor prices, demand forecast
    Memory: Past promotions, market trends

EDUCATION:
==========
- Personalized tutoring
- Curriculum planning
- Assessment creation
- Learning path optimization

Example Agent:
    Task: "Create personalized study plan for calculus"
    Tools: Curriculum database, assessment generator
    Memory: Student progress, learning style
    """)


# =============================================================================
# SECTION 6: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common AI agents interview questions"""
    print("\n" + "=" * 70)
    print("AI AGENTS INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is the ReAct pattern?",
         "Reasoning and Acting interleaved. Agent alternates between thinking (reasoning about what to do) and acting (using tools). Format: Thought -> Action -> Observation -> repeat. Improves interpretability and reliability."),
        ("How do you handle agent failures?",
         "Strategies: Retry with different approach, fallback to simpler method, ask for human help, graceful degradation. Important: Log failures, set max iterations, have timeout."),
        ("What's the difference between agents and chains?",
         "Chains: Fixed sequence of steps, deterministic. Agents: Dynamic, LLM decides next action based on observations. Use chains for predictable workflows, agents for open-ended tasks."),
        ("How do you evaluate agent performance?",
         "Metrics: Task completion rate, steps to completion, tool usage efficiency, cost. Also: Human evaluation of quality, safety checks, edge case handling."),
        ("What are the risks of autonomous agents?",
         "Risks: Unintended actions, infinite loops, cost explosion, security vulnerabilities. Mitigations: Human oversight, sandboxing, rate limits, approval for sensitive actions."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 7: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common agent mistakes"""
    print("\n" + "=" * 70)
    print("COMMON AGENT PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Infinite loops", "Set max iterations, detect repetition, add escape conditions"),
        ("Tool misuse", "Clear tool descriptions, validate inputs, handle errors gracefully"),
        ("Context overflow", "Summarize history, use RAG for long-term memory"),
        ("Over-autonomy", "Add human checkpoints for important decisions"),
        ("Poor error handling", "Catch exceptions, provide fallbacks, log for debugging"),
        ("Cost explosion", "Set budgets, monitor usage, cache repeated calls"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 17: AI AGENTS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = AgentAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What are AI Agents? (4W+H)")
        print("  2. Agent Architectures")
        print("  3. Tools and Capabilities")
        print("  4. Memory Systems")
        print("  5. Multi-Domain Use Cases")
        print()
        print("DEEP DIVE:")
        print("  6. Interview Questions")
        print("  7. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  8. AI Explanation (OpenAI)")
        print("  9. AI Explanation (Ollama - FREE)")
        print("  10. Run Simple Agent Demo (OpenAI)")
        print()
        print("  11. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-11): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_agents_comprehensive()
        elif choice == "2":
            agent_architectures()
        elif choice == "3":
            tools_and_capabilities()
        elif choice == "4":
            memory_systems()
        elif choice == "5":
            multi_domain_use_cases()
        elif choice == "6":
            interview_questions()
        elif choice == "7":
            common_pitfalls()
        elif choice == "8":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "9":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "10":
            task = input("Enter task for agent: ").strip()
            print("\nAgent Response:")
            print(assistant.run_simple_agent(task))
        elif choice == "11":
            explain_agents_comprehensive()
            agent_architectures()
            tools_and_capabilities()
            memory_systems()
            multi_domain_use_cases()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
