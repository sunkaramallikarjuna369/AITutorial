import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Bot, Brain, Workflow, Code, Zap, MessageSquare, Search, FileText, Stethoscope, Briefcase, ShoppingCart, GraduationCap, Wrench } from 'lucide-react'

const Module17Agents = () => {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [agentSteps, setAgentSteps] = useState<string[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const agentTypes = [
    {
      id: 'react',
      name: 'ReAct Agent',
      icon: Brain,
      description: 'Reason + Act: Think step by step, then take action',
      color: 'bg-purple-500',
      steps: ['Observe the problem', 'Think about what to do', 'Take an action', 'Observe the result', 'Repeat until solved']
    },
    {
      id: 'tool',
      name: 'Tool-Using Agent',
      icon: Wrench,
      description: 'Uses external tools like search, calculator, APIs',
      color: 'bg-blue-500',
      steps: ['Receive user query', 'Decide which tool to use', 'Call the tool', 'Process tool output', 'Generate response']
    },
    {
      id: 'planning',
      name: 'Planning Agent',
      icon: Workflow,
      description: 'Creates and executes multi-step plans',
      color: 'bg-green-500',
      steps: ['Understand the goal', 'Break into subtasks', 'Order the subtasks', 'Execute each step', 'Verify completion']
    },
    {
      id: 'multi',
      name: 'Multi-Agent System',
      icon: Bot,
      description: 'Multiple specialized agents working together',
      color: 'bg-orange-500',
      steps: ['Coordinator receives task', 'Assigns to specialist agents', 'Agents work in parallel', 'Results are combined', 'Final answer delivered']
    }
  ]

  const useCases = [
    {
      domain: 'Healthcare',
      icon: Stethoscope,
      color: 'bg-red-500',
      examples: [
        { name: 'Medical Research Agent', desc: 'Searches medical literature, summarizes findings, suggests treatments' },
        { name: 'Patient Triage Agent', desc: 'Assesses symptoms, prioritizes cases, routes to specialists' },
        { name: 'Drug Interaction Checker', desc: 'Analyzes medications, identifies conflicts, suggests alternatives' },
        { name: 'Clinical Trial Matcher', desc: 'Matches patients to relevant clinical trials based on criteria' }
      ]
    },
    {
      domain: 'Finance',
      icon: Briefcase,
      color: 'bg-green-500',
      examples: [
        { name: 'Trading Agent', desc: 'Analyzes markets, executes trades, manages portfolio risk' },
        { name: 'Fraud Detection Agent', desc: 'Monitors transactions, identifies anomalies, blocks suspicious activity' },
        { name: 'Financial Advisor Agent', desc: 'Assesses goals, recommends investments, rebalances portfolios' },
        { name: 'Loan Processing Agent', desc: 'Evaluates applications, verifies documents, calculates risk scores' }
      ]
    },
    {
      domain: 'Customer Service',
      icon: MessageSquare,
      color: 'bg-blue-500',
      examples: [
        { name: 'Support Agent', desc: 'Handles inquiries, resolves issues, escalates when needed' },
        { name: 'Order Management Agent', desc: 'Tracks orders, processes returns, handles refunds' },
        { name: 'Feedback Analysis Agent', desc: 'Analyzes reviews, identifies trends, suggests improvements' },
        { name: 'Appointment Scheduler', desc: 'Books meetings, sends reminders, handles rescheduling' }
      ]
    },
    {
      domain: 'Research',
      icon: Search,
      color: 'bg-purple-500',
      examples: [
        { name: 'Literature Review Agent', desc: 'Searches papers, extracts key findings, synthesizes knowledge' },
        { name: 'Data Analysis Agent', desc: 'Processes datasets, runs statistics, generates visualizations' },
        { name: 'Hypothesis Generator', desc: 'Identifies gaps in research, proposes new experiments' },
        { name: 'Citation Manager', desc: 'Organizes references, formats citations, checks for plagiarism' }
      ]
    },
    {
      domain: 'E-Commerce',
      icon: ShoppingCart,
      color: 'bg-yellow-500',
      examples: [
        { name: 'Product Recommendation Agent', desc: 'Analyzes preferences, suggests products, personalizes experience' },
        { name: 'Inventory Management Agent', desc: 'Tracks stock, predicts demand, automates reordering' },
        { name: 'Price Optimization Agent', desc: 'Monitors competitors, adjusts pricing, maximizes revenue' },
        { name: 'Review Moderation Agent', desc: 'Filters spam, verifies purchases, highlights helpful reviews' }
      ]
    },
    {
      domain: 'Education',
      icon: GraduationCap,
      color: 'bg-indigo-500',
      examples: [
        { name: 'Tutoring Agent', desc: 'Explains concepts, answers questions, adapts to learning style' },
        { name: 'Assignment Grader', desc: 'Evaluates submissions, provides feedback, detects plagiarism' },
        { name: 'Curriculum Designer', desc: 'Creates lesson plans, sequences topics, aligns with standards' },
        { name: 'Study Buddy Agent', desc: 'Creates flashcards, quizzes students, tracks progress' }
      ]
    }
  ]

  const simulateAgent = (agentId: string) => {
    const agent = agentTypes.find(a => a.id === agentId)
    if (!agent) return

    setSelectedAgent(agentId)
    setAgentSteps([])
    setIsRunning(true)

    agent.steps.forEach((step, index) => {
      setTimeout(() => {
        setAgentSteps(prev => [...prev, step])
        if (index === agent.steps.length - 1) {
          setIsRunning(false)
        }
      }, (index + 1) * 800)
    })
  }

  const pythonCode = `"""
AI Agents - Autonomous AI Systems
=================================
Build intelligent agents that can reason, plan, and use tools!
"""

# ============================================
# 1. Simple ReAct Agent (Reason + Act)
# ============================================

class ReActAgent:
    """
    ReAct Agent: Thinks step by step, then acts
    
    Pattern: Thought -> Action -> Observation -> Repeat
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def think(self, observation):
        """Generate a thought based on observation"""
        prompt = f"""
        Previous observations: {self.memory}
        Current observation: {observation}
        
        Think step by step about what to do next.
        """
        thought = self.llm.generate(prompt)
        return thought
    
    def act(self, thought):
        """Decide and execute an action"""
        prompt = f"""
        Thought: {thought}
        Available tools: {list(self.tools.keys())}
        
        Which tool should I use and with what input?
        """
        action = self.llm.generate(prompt)
        
        # Parse and execute action
        tool_name, tool_input = self.parse_action(action)
        result = self.tools[tool_name](tool_input)
        return result
    
    def run(self, query, max_steps=5):
        """Run the agent loop"""
        observation = query
        
        for step in range(max_steps):
            print(f"\\n--- Step {step + 1} ---")
            
            # Think
            thought = self.think(observation)
            print(f"Thought: {thought}")
            
            # Check if done
            if "FINAL ANSWER" in thought:
                return thought
            
            # Act
            observation = self.act(thought)
            print(f"Observation: {observation}")
            
            self.memory.append({
                'thought': thought,
                'observation': observation
            })
        
        return "Max steps reached"


# ============================================
# 2. Tool-Using Agent with LangChain
# ============================================

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

def create_tool_agent():
    """Create an agent that can use multiple tools"""
    
    # Initialize LLM
    llm = OpenAI(temperature=0)
    
    # Define tools
    search = DuckDuckGoSearchRun()
    
    def calculator(expression):
        """Evaluate mathematical expressions"""
        try:
            return str(eval(expression))
        except:
            return "Error in calculation"
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Search the internet for current information"
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="Perform mathematical calculations"
        )
    ]
    
    # Create agent
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    
    return agent

# Usage
# agent = create_tool_agent()
# result = agent.run("What is the population of France divided by 1000?")


# ============================================
# 3. Planning Agent with LangGraph
# ============================================

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class PlannerState(TypedDict):
    goal: str
    plan: List[str]
    current_step: int
    results: List[str]
    final_answer: str

def create_planning_agent():
    """Create an agent that plans before acting"""
    
    def planner(state):
        """Create a plan to achieve the goal"""
        goal = state['goal']
        
        # Use LLM to create plan
        plan = [
            "Research the topic",
            "Gather relevant data",
            "Analyze the information",
            "Synthesize findings",
            "Generate final answer"
        ]
        
        return {'plan': plan, 'current_step': 0}
    
    def executor(state):
        """Execute the current step"""
        step = state['plan'][state['current_step']]
        
        # Execute step (simplified)
        result = f"Completed: {step}"
        
        return {
            'results': state['results'] + [result],
            'current_step': state['current_step'] + 1
        }
    
    def should_continue(state):
        """Check if we should continue or finish"""
        if state['current_step'] >= len(state['plan']):
            return 'finish'
        return 'execute'
    
    def finisher(state):
        """Generate final answer"""
        return {'final_answer': f"Completed all {len(state['plan'])} steps"}
    
    # Build graph
    workflow = StateGraph(PlannerState)
    workflow.add_node("plan", planner)
    workflow.add_node("execute", executor)
    workflow.add_node("finish", finisher)
    
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    workflow.add_conditional_edges("execute", should_continue, {
        'execute': 'execute',
        'finish': 'finish'
    })
    workflow.add_edge("finish", END)
    
    return workflow.compile()


# ============================================
# 4. Multi-Agent System with AutoGen
# ============================================

import autogen

def create_multi_agent_system():
    """Create a team of specialized agents"""
    
    config_list = [{"model": "gpt-4", "api_key": "your-key"}]
    
    # Create specialized agents
    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message="You are a research specialist. Find and analyze information.",
        llm_config={"config_list": config_list}
    )
    
    writer = autogen.AssistantAgent(
        name="Writer",
        system_message="You are a writing specialist. Create clear, engaging content.",
        llm_config={"config_list": config_list}
    )
    
    critic = autogen.AssistantAgent(
        name="Critic",
        system_message="You review work and provide constructive feedback.",
        llm_config={"config_list": config_list}
    )
    
    # Create user proxy (coordinator)
    user_proxy = autogen.UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "workspace"}
    )
    
    # Create group chat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, writer, critic],
        messages=[],
        max_round=10
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list}
    )
    
    return user_proxy, manager

# Usage
# user_proxy, manager = create_multi_agent_system()
# user_proxy.initiate_chat(manager, message="Write a blog post about AI agents")


# ============================================
# 5. Healthcare Agent Example
# ============================================

class HealthcareAgent:
    """Agent for medical research and patient support"""
    
    def __init__(self, llm, medical_db):
        self.llm = llm
        self.medical_db = medical_db
    
    def search_literature(self, query):
        """Search medical literature"""
        results = self.medical_db.search(query)
        return self.summarize_findings(results)
    
    def check_drug_interactions(self, medications):
        """Check for drug interactions"""
        interactions = []
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                interaction = self.medical_db.check_interaction(med1, med2)
                if interaction:
                    interactions.append(interaction)
        return interactions
    
    def triage_patient(self, symptoms):
        """Assess patient symptoms and prioritize"""
        prompt = f"""
        Patient symptoms: {symptoms}
        
        Assess urgency (1-5) and recommend next steps.
        Consider: severity, potential conditions, recommended specialists.
        """
        assessment = self.llm.generate(prompt)
        return assessment


# ============================================
# 6. Financial Trading Agent
# ============================================

class TradingAgent:
    """Autonomous trading agent with risk management"""
    
    def __init__(self, llm, market_api, risk_limit=0.02):
        self.llm = llm
        self.market = market_api
        self.risk_limit = risk_limit
        self.portfolio = {}
    
    def analyze_market(self, symbol):
        """Analyze market conditions"""
        data = self.market.get_data(symbol)
        
        analysis = {
            'trend': self.detect_trend(data),
            'volatility': self.calculate_volatility(data),
            'sentiment': self.analyze_sentiment(symbol)
        }
        return analysis
    
    def make_decision(self, symbol):
        """Decide whether to buy, sell, or hold"""
        analysis = self.analyze_market(symbol)
        
        prompt = f"""
        Market analysis for {symbol}:
        - Trend: {analysis['trend']}
        - Volatility: {analysis['volatility']}
        - Sentiment: {analysis['sentiment']}
        
        Current portfolio: {self.portfolio}
        Risk limit: {self.risk_limit}
        
        Recommend: BUY, SELL, or HOLD with reasoning.
        """
        
        decision = self.llm.generate(prompt)
        return decision
    
    def execute_trade(self, symbol, action, amount):
        """Execute trade with risk checks"""
        if not self.check_risk(symbol, action, amount):
            return "Trade rejected: exceeds risk limit"
        
        result = self.market.execute(symbol, action, amount)
        self.update_portfolio(symbol, action, amount)
        return result


# ============================================
# 7. Customer Service Agent
# ============================================

class CustomerServiceAgent:
    """Multi-capability customer service agent"""
    
    def __init__(self, llm, knowledge_base, order_system):
        self.llm = llm
        self.kb = knowledge_base
        self.orders = order_system
        self.conversation_history = []
    
    def handle_inquiry(self, message, customer_id):
        """Handle customer inquiry"""
        # Classify intent
        intent = self.classify_intent(message)
        
        if intent == 'order_status':
            return self.check_order_status(customer_id)
        elif intent == 'return_request':
            return self.process_return(customer_id, message)
        elif intent == 'product_question':
            return self.answer_product_question(message)
        elif intent == 'complaint':
            return self.handle_complaint(customer_id, message)
        else:
            return self.general_response(message)
    
    def classify_intent(self, message):
        """Classify customer intent"""
        prompt = f"""
        Customer message: {message}
        
        Classify intent: order_status, return_request, 
        product_question, complaint, or general
        """
        return self.llm.generate(prompt).strip().lower()
    
    def should_escalate(self, message, sentiment_score):
        """Determine if human escalation is needed"""
        escalation_triggers = [
            sentiment_score < -0.5,  # Very negative sentiment
            'speak to human' in message.lower(),
            'manager' in message.lower(),
            len(self.conversation_history) > 5  # Long conversation
        ]
        return any(escalation_triggers)


# ============================================
# 8. Research Assistant Agent
# ============================================

class ResearchAgent:
    """Agent for academic research assistance"""
    
    def __init__(self, llm, paper_db, citation_manager):
        self.llm = llm
        self.papers = paper_db
        self.citations = citation_manager
    
    def literature_review(self, topic, num_papers=20):
        """Conduct automated literature review"""
        # Search for papers
        papers = self.papers.search(topic, limit=num_papers)
        
        # Extract key findings
        findings = []
        for paper in papers:
            summary = self.summarize_paper(paper)
            findings.append({
                'title': paper['title'],
                'authors': paper['authors'],
                'year': paper['year'],
                'key_findings': summary
            })
        
        # Synthesize findings
        synthesis = self.synthesize_findings(findings)
        
        return {
            'papers_reviewed': len(papers),
            'findings': findings,
            'synthesis': synthesis,
            'gaps': self.identify_gaps(findings)
        }
    
    def generate_hypothesis(self, research_area):
        """Generate research hypotheses"""
        # Get current state of research
        current_knowledge = self.literature_review(research_area, num_papers=10)
        
        prompt = f"""
        Research area: {research_area}
        Current knowledge: {current_knowledge['synthesis']}
        Identified gaps: {current_knowledge['gaps']}
        
        Generate 3 novel, testable research hypotheses.
        """
        
        hypotheses = self.llm.generate(prompt)
        return hypotheses


# ============================================
# 9. Coding Assistant Agent
# ============================================

class CodingAgent:
    """Agent that can write, debug, and explain code"""
    
    def __init__(self, llm, code_executor):
        self.llm = llm
        self.executor = code_executor
    
    def write_code(self, specification):
        """Generate code from specification"""
        prompt = f"""
        Write Python code for: {specification}
        
        Include:
        - Clear function/class structure
        - Type hints
        - Docstrings
        - Error handling
        """
        code = self.llm.generate(prompt)
        return code
    
    def debug_code(self, code, error):
        """Debug code given an error"""
        prompt = f"""
        Code:
        {code}
        
        Error:
        {error}
        
        Identify the bug and provide fixed code.
        """
        fix = self.llm.generate(prompt)
        return fix
    
    def explain_code(self, code):
        """Explain code in simple terms"""
        prompt = f"""
        Explain this code in simple terms that a beginner can understand:
        
        {code}
        
        Include:
        - What the code does overall
        - Step-by-step explanation
        - Key concepts used
        """
        explanation = self.llm.generate(prompt)
        return explanation
    
    def run_and_iterate(self, specification, max_attempts=3):
        """Write code, test it, fix errors iteratively"""
        code = self.write_code(specification)
        
        for attempt in range(max_attempts):
            result = self.executor.run(code)
            
            if result['success']:
                return {'code': code, 'output': result['output']}
            
            # Debug and retry
            code = self.debug_code(code, result['error'])
        
        return {'code': code, 'error': 'Max attempts reached'}


# ============================================
# 10. E-Commerce Agent
# ============================================

class ECommerceAgent:
    """Agent for e-commerce operations"""
    
    def __init__(self, llm, product_db, user_db):
        self.llm = llm
        self.products = product_db
        self.users = user_db
    
    def recommend_products(self, user_id):
        """Generate personalized recommendations"""
        user = self.users.get(user_id)
        
        # Get user history and preferences
        history = user['purchase_history']
        preferences = user['preferences']
        
        # Find similar products
        candidates = self.products.find_similar(history)
        
        # Rank by relevance
        prompt = f"""
        User preferences: {preferences}
        Purchase history: {history}
        Candidate products: {candidates}
        
        Rank top 5 products with explanations.
        """
        
        recommendations = self.llm.generate(prompt)
        return recommendations
    
    def optimize_pricing(self, product_id):
        """Optimize product pricing"""
        product = self.products.get(product_id)
        competitors = self.products.get_competitor_prices(product_id)
        demand = self.products.get_demand_curve(product_id)
        
        prompt = f"""
        Product: {product['name']}
        Current price: {product['price']}
        Competitor prices: {competitors}
        Demand elasticity: {demand}
        
        Recommend optimal price to maximize revenue.
        """
        
        recommendation = self.llm.generate(prompt)
        return recommendation


print("AI Agents module loaded!")
print("Explore different agent architectures and use cases.")
`

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center gap-3">
          <Bot className="w-10 h-10 text-purple-400" />
          AI Agents
        </h1>
        <p className="text-xl text-slate-300">
          Autonomous AI systems that can reason, plan, and take actions!
        </p>
      </div>

      <Tabs defaultValue="learn" className="w-full">
        <TabsList className="grid w-full grid-cols-3 mb-6">
          <TabsTrigger value="learn">Learn</TabsTrigger>
          <TabsTrigger value="visualize">Visualize</TabsTrigger>
          <TabsTrigger value="code">Code</TabsTrigger>
        </TabsList>

        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Brain className="w-6 h-6 text-purple-400" />
                What are AI Agents?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-purple-400">AI Agents</strong> are like smart robots that can think and act on their own! 
                Instead of just answering questions, they can actually DO things - search the internet, 
                write code, send emails, and much more!
              </p>
              
              <div className="bg-slate-900 p-4 rounded-lg">
                <h4 className="text-white font-semibold mb-2">Think of it like this:</h4>
                <p>
                  <strong className="text-blue-400">Regular AI (ChatGPT):</strong> Like a smart friend who gives advice
                  <br />
                  <strong className="text-purple-400">AI Agent:</strong> Like a smart assistant who actually does the work for you!
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-green-400 font-semibold mb-2">Key Components</h4>
                  <ul className="space-y-1 text-sm">
                    <li>🧠 <strong>Brain (LLM):</strong> Thinks and makes decisions</li>
                    <li>🔧 <strong>Tools:</strong> Search, calculator, APIs, etc.</li>
                    <li>💾 <strong>Memory:</strong> Remembers past actions</li>
                    <li>📋 <strong>Planning:</strong> Creates step-by-step plans</li>
                  </ul>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-yellow-400 font-semibold mb-2">Agent Loop</h4>
                  <ul className="space-y-1 text-sm">
                    <li>1️⃣ <strong>Observe:</strong> See the current situation</li>
                    <li>2️⃣ <strong>Think:</strong> Reason about what to do</li>
                    <li>3️⃣ <strong>Act:</strong> Take an action using tools</li>
                    <li>4️⃣ <strong>Repeat:</strong> Until goal is achieved</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Workflow className="w-6 h-6 text-blue-400" />
                Types of AI Agents
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {agentTypes.map((agent) => (
                  <div key={agent.id} className="bg-slate-900 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`p-2 rounded-lg ${agent.color}`}>
                        <agent.icon className="w-5 h-5 text-white" />
                      </div>
                      <h4 className="text-white font-semibold">{agent.name}</h4>
                    </div>
                    <p className="text-slate-400 text-sm">{agent.description}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Zap className="w-6 h-6 text-yellow-400" />
                Popular Agent Frameworks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { name: 'LangChain', desc: 'Most popular framework', color: 'bg-green-500' },
                  { name: 'AutoGPT', desc: 'Fully autonomous', color: 'bg-purple-500' },
                  { name: 'CrewAI', desc: 'Multi-agent teams', color: 'bg-blue-500' },
                  { name: 'LangGraph', desc: 'Stateful workflows', color: 'bg-orange-500' },
                  { name: 'AutoGen', desc: 'Microsoft framework', color: 'bg-cyan-500' },
                  { name: 'AgentGPT', desc: 'Web-based agents', color: 'bg-pink-500' },
                  { name: 'BabyAGI', desc: 'Task-driven agents', color: 'bg-yellow-500' },
                  { name: 'SuperAGI', desc: 'Production agents', color: 'bg-red-500' }
                ].map((framework) => (
                  <div key={framework.name} className="bg-slate-900 p-3 rounded-lg text-center">
                    <Badge className={`${framework.color} mb-2`}>{framework.name}</Badge>
                    <p className="text-slate-400 text-xs">{framework.desc}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Bot className="w-6 h-6 text-purple-400" />
                Interactive Agent Simulator
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click on an agent type to see how it works step by step!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {agentTypes.map((agent) => (
                  <Button
                    key={agent.id}
                    onClick={() => simulateAgent(agent.id)}
                    disabled={isRunning}
                    className={`h-auto py-4 flex flex-col items-center gap-2 ${
                      selectedAgent === agent.id ? agent.color : 'bg-slate-700 hover:bg-slate-600'
                    }`}
                  >
                    <agent.icon className="w-8 h-8" />
                    <span className="text-sm">{agent.name}</span>
                  </Button>
                ))}
              </div>

              {selectedAgent && (
                <div className="bg-slate-900 p-6 rounded-lg">
                  <h4 className="text-white font-semibold mb-4">
                    {agentTypes.find(a => a.id === selectedAgent)?.name} in Action:
                  </h4>
                  <div className="space-y-3">
                    {agentSteps.map((step, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-3 text-slate-300 animate-fade-in"
                      >
                        <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-white font-bold">
                          {index + 1}
                        </div>
                        <span>{step}</span>
                        {index === agentSteps.length - 1 && !isRunning && (
                          <Badge className="bg-green-500 ml-auto">Complete!</Badge>
                        )}
                      </div>
                    ))}
                    {isRunning && (
                      <div className="flex items-center gap-2 text-purple-400">
                        <div className="animate-spin w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full" />
                        Processing...
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Briefcase className="w-6 h-6 text-green-400" />
                Real-World Use Cases by Domain
              </CardTitle>
              <CardDescription className="text-slate-400">
                Explore how AI agents are used across different industries
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {useCases.map((useCase) => (
                  <div
                    key={useCase.domain}
                    className="bg-slate-900 p-4 rounded-lg cursor-pointer hover:ring-2 hover:ring-purple-500 transition-all"
                    onClick={() => setSelectedAgent(useCase.domain)}
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <div className={`p-2 rounded-lg ${useCase.color}`}>
                        <useCase.icon className="w-5 h-5 text-white" />
                      </div>
                      <h4 className="text-white font-semibold">{useCase.domain}</h4>
                    </div>
                    <div className="space-y-2">
                      {useCase.examples.slice(0, 2).map((example, idx) => (
                        <div key={idx} className="text-xs">
                          <span className="text-purple-400 font-medium">{example.name}</span>
                          <p className="text-slate-500">{example.desc}</p>
                        </div>
                      ))}
                      <p className="text-slate-600 text-xs">+{useCase.examples.length - 2} more...</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Workflow className="w-6 h-6 text-blue-400" />
                Agent Architecture Diagram
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-900 p-6 rounded-lg">
                <svg viewBox="0 0 800 400" className="w-full h-64">
                  {/* User */}
                  <rect x="50" y="160" width="100" height="60" rx="10" fill="#6366f1" />
                  <text x="100" y="195" textAnchor="middle" fill="white" fontSize="14">User</text>
                  
                  {/* Agent Brain */}
                  <rect x="250" y="100" width="140" height="180" rx="10" fill="#8b5cf6" />
                  <text x="320" y="130" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">Agent</text>
                  <rect x="265" y="145" width="110" height="30" rx="5" fill="#a78bfa" />
                  <text x="320" y="165" textAnchor="middle" fill="white" fontSize="11">LLM Brain</text>
                  <rect x="265" y="185" width="110" height="30" rx="5" fill="#a78bfa" />
                  <text x="320" y="205" textAnchor="middle" fill="white" fontSize="11">Memory</text>
                  <rect x="265" y="225" width="110" height="30" rx="5" fill="#a78bfa" />
                  <text x="320" y="245" textAnchor="middle" fill="white" fontSize="11">Planner</text>
                  
                  {/* Tools */}
                  <rect x="500" y="50" width="100" height="40" rx="5" fill="#22c55e" />
                  <text x="550" y="75" textAnchor="middle" fill="white" fontSize="11">Search</text>
                  
                  <rect x="500" y="110" width="100" height="40" rx="5" fill="#22c55e" />
                  <text x="550" y="135" textAnchor="middle" fill="white" fontSize="11">Calculator</text>
                  
                  <rect x="500" y="170" width="100" height="40" rx="5" fill="#22c55e" />
                  <text x="550" y="195" textAnchor="middle" fill="white" fontSize="11">Code Exec</text>
                  
                  <rect x="500" y="230" width="100" height="40" rx="5" fill="#22c55e" />
                  <text x="550" y="255" textAnchor="middle" fill="white" fontSize="11">APIs</text>
                  
                  <rect x="500" y="290" width="100" height="40" rx="5" fill="#22c55e" />
                  <text x="550" y="315" textAnchor="middle" fill="white" fontSize="11">Database</text>
                  
                  {/* Arrows */}
                  <path d="M150 190 L250 190" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" />
                  <path d="M390 190 L500 70" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" />
                  <path d="M390 190 L500 130" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" />
                  <path d="M390 190 L500 190" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" />
                  <path d="M390 190 L500 250" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" />
                  <path d="M390 190 L500 310" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" />
                  
                  {/* Labels */}
                  <text x="200" y="180" textAnchor="middle" fill="#94a3b8" fontSize="10">Query</text>
                  <text x="440" y="120" textAnchor="middle" fill="#94a3b8" fontSize="10">Tools</text>
                  
                  <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                    </marker>
                  </defs>
                </svg>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-6 h-6 text-green-400" />
                AI Agents Python Code
              </CardTitle>
              <CardDescription className="text-slate-400">
                Complete implementations of various agent types and use cases
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-900 p-4 rounded-lg overflow-x-auto">
                <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono">
                  {pythonCode}
                </pre>
              </div>
              <div className="mt-4 flex gap-2">
                <Button
                  onClick={() => navigator.clipboard.writeText(pythonCode)}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Copy Code
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Required Libraries</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-900 p-4 rounded-lg">
                <pre className="text-sm text-green-400 font-mono">
{`# Install required packages
pip install langchain langchain-openai langgraph
pip install autogen-agentchat
pip install crewai
pip install duckduckgo-search`}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module17Agents
