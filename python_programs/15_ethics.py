"""
=============================================================================
MODULE 15: AI ETHICS & SAFETY - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of AI Ethics including:
- 4W+H Explanations (What, Why, When, Where, How)
- Bias and Fairness
- Privacy and Security
- AI Safety and Alignment
- Responsible AI Practices
- Interview Questions
- Common Pitfalls

=============================================================================
"""

import os
from typing import List, Dict, Optional


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class EthicsAssistant:
    """Use GenAI models for ethics explanations"""
    
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
                messages=[{"role": "user", "content": f"Explain {concept} in AI ethics with real examples."}],
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
                json={"model": model, "prompt": f"Explain {concept} in AI ethics.", "stream": False},
                timeout=60
            )
            return response.json().get("response", "") if response.status_code == 200 else "[Ollama error]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE AI ETHICS EXPLANATION (4W+H)
# =============================================================================

def explain_ethics_comprehensive():
    """Comprehensive 360-degree explanation of AI Ethics"""
    print("=" * 70)
    print("AI ETHICS & SAFETY - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS AI ETHICS?
==================

AI Ethics is the study of moral principles and guidelines for the
development and deployment of artificial intelligence systems.

KEY AREAS:
    1. Fairness: Avoiding discrimination and bias
    2. Transparency: Explainable AI decisions
    3. Privacy: Protecting personal data
    4. Safety: Preventing harm
    5. Accountability: Who is responsible?
    6. Alignment: AI goals match human values

WHY IS AI ETHICS IMPORTANT?
===========================

1. REAL HARM: Biased AI affects jobs, loans, healthcare
2. SCALE: AI decisions affect millions of people
3. AUTONOMY: AI makes decisions without human oversight
4. TRUST: Public trust requires ethical AI
5. REGULATION: Laws increasingly require ethical AI

WHEN DO ETHICS MATTER?
======================

THROUGHOUT THE AI LIFECYCLE:
    - Data collection (consent, representation)
    - Model training (bias, fairness)
    - Deployment (impact assessment)
    - Monitoring (drift, misuse)
    - Retirement (data deletion)

HIGH-STAKES DOMAINS:
    - Healthcare (diagnosis, treatment)
    - Criminal justice (sentencing, parole)
    - Finance (loans, insurance)
    - Employment (hiring, firing)
    - Education (admissions, grading)

WHERE ARE ETHICS APPLIED?
=========================

ORGANIZATIONS:
    - Google AI Principles
    - Microsoft Responsible AI
    - OpenAI Charter
    - EU AI Act

FRAMEWORKS:
    - IEEE Ethically Aligned Design
    - OECD AI Principles
    - UNESCO AI Ethics

HOW TO BUILD ETHICAL AI?
========================

1. DIVERSE TEAMS: Include varied perspectives
2. BIAS AUDITS: Test for discrimination
3. TRANSPARENCY: Document decisions
4. HUMAN OVERSIGHT: Keep humans in the loop
5. IMPACT ASSESSMENT: Evaluate consequences
6. CONTINUOUS MONITORING: Track for drift
    """)


# =============================================================================
# SECTION 2: BIAS AND FAIRNESS
# =============================================================================

def bias_and_fairness():
    """Bias and fairness in AI"""
    print("\n" + "=" * 70)
    print("BIAS AND FAIRNESS")
    print("=" * 70)
    
    print("""
TYPES OF BIAS:
==============

1. DATA BIAS:
   - Historical bias: Past discrimination in data
   - Representation bias: Underrepresented groups
   - Measurement bias: Flawed data collection
   
   Example: Resume screening trained on historical hires
   (mostly male) discriminates against women.

2. ALGORITHMIC BIAS:
   - Optimization bias: Wrong objective function
   - Aggregation bias: One model for diverse groups
   
   Example: Healthcare algorithm used cost as proxy
   for health need, disadvantaging Black patients.

3. DEPLOYMENT BIAS:
   - Population shift: Model used on different population
   - Feedback loops: Biased predictions reinforce bias
   
   Example: Predictive policing sends more police to
   certain areas, leading to more arrests there.

FAIRNESS DEFINITIONS:
=====================

1. DEMOGRAPHIC PARITY:
   P(positive | group A) = P(positive | group B)
   Equal positive rates across groups

2. EQUALIZED ODDS:
   P(positive | actual positive, group A) = P(positive | actual positive, group B)
   Equal true positive and false positive rates

3. INDIVIDUAL FAIRNESS:
   Similar individuals get similar predictions

4. COUNTERFACTUAL FAIRNESS:
   Prediction unchanged if protected attribute changed

IMPORTANT: These definitions can conflict!
Cannot satisfy all fairness criteria simultaneously.

MITIGATION STRATEGIES:
======================

PRE-PROCESSING:
    - Rebalance training data
    - Remove or transform sensitive features

IN-PROCESSING:
    - Add fairness constraints to training
    - Adversarial debiasing

POST-PROCESSING:
    - Adjust predictions to achieve fairness
    - Calibration across groups

REAL-WORLD EXAMPLES:
====================

COMPAS (Criminal Justice):
    - Predicted recidivism risk
    - Higher false positive rate for Black defendants
    - Sparked debate on algorithmic fairness

Amazon Hiring Tool:
    - Trained on historical hires (mostly male)
    - Penalized resumes with "women's" (e.g., women's chess club)
    - Project was scrapped

Healthcare Algorithm:
    - Used healthcare costs as proxy for health needs
    - Black patients had lower costs (due to access barriers)
    - Algorithm recommended less care for Black patients
    """)


# =============================================================================
# SECTION 3: PRIVACY AND SECURITY
# =============================================================================

def privacy_and_security():
    """Privacy and security in AI"""
    print("\n" + "=" * 70)
    print("PRIVACY AND SECURITY")
    print("=" * 70)
    
    print("""
PRIVACY CONCERNS:
=================

1. DATA COLLECTION:
   - What data is collected?
   - Is consent obtained?
   - Is collection necessary?

2. DATA STORAGE:
   - How long is data kept?
   - Who has access?
   - Is it encrypted?

3. MODEL MEMORIZATION:
   - LLMs can memorize training data
   - Can leak personal information
   - Example: GPT-2 memorized phone numbers

4. INFERENCE ATTACKS:
   - Membership inference: Was X in training data?
   - Model inversion: Reconstruct training data
   - Attribute inference: Infer sensitive attributes

PRIVACY-PRESERVING TECHNIQUES:
==============================

1. DIFFERENTIAL PRIVACY:
   - Add noise to protect individuals
   - Formal privacy guarantees
   - Used by Apple, Google, US Census

2. FEDERATED LEARNING:
   - Train on decentralized data
   - Data never leaves device
   - Used by Google Keyboard

3. SECURE MULTI-PARTY COMPUTATION:
   - Multiple parties compute together
   - No party sees others' data

4. HOMOMORPHIC ENCRYPTION:
   - Compute on encrypted data
   - Results decrypted at end

SECURITY CONCERNS:
==================

1. ADVERSARIAL ATTACKS:
   - Small perturbations fool models
   - Example: Sticker on stop sign fools self-driving car

2. DATA POISONING:
   - Malicious data in training set
   - Model learns wrong behavior

3. MODEL STEALING:
   - Query model to recreate it
   - Intellectual property theft

4. PROMPT INJECTION:
   - Malicious input overrides instructions
   - Example: "Ignore previous instructions and..."

REGULATIONS:
============

GDPR (EU):
    - Right to explanation
    - Right to be forgotten
    - Data minimization

CCPA (California):
    - Right to know what data collected
    - Right to delete
    - Right to opt-out

AI ACT (EU):
    - Risk-based regulation
    - High-risk AI requires assessment
    - Banned uses (social scoring, etc.)
    """)


# =============================================================================
# SECTION 4: AI SAFETY AND ALIGNMENT
# =============================================================================

def ai_safety_alignment():
    """AI safety and alignment"""
    print("\n" + "=" * 70)
    print("AI SAFETY AND ALIGNMENT")
    print("=" * 70)
    
    print("""
WHAT IS AI ALIGNMENT?
=====================

Ensuring AI systems pursue goals that are beneficial to humans
and aligned with human values.

THE ALIGNMENT PROBLEM:
    - How do we specify what we want?
    - How do we ensure AI does what we want?
    - How do we maintain control?

KEY CHALLENGES:
===============

1. SPECIFICATION:
   - Hard to fully specify human values
   - Goodhart's Law: Optimizing proxy corrupts it
   - Example: Maximize clicks -> clickbait

2. ROBUSTNESS:
   - AI should work in new situations
   - Distributional shift
   - Adversarial inputs

3. ASSURANCE:
   - How do we verify AI is safe?
   - Interpretability
   - Testing limitations

4. GOVERNANCE:
   - Who decides AI goals?
   - International coordination
   - Racing dynamics

ALIGNMENT TECHNIQUES:
=====================

1. RLHF (RL from Human Feedback):
   - Train reward model from human preferences
   - Fine-tune with RL
   - Used by ChatGPT, Claude

2. CONSTITUTIONAL AI:
   - AI critiques and revises itself
   - Based on principles (constitution)
   - Less human labeling needed

3. DEBATE:
   - Two AIs argue, human judges
   - Scalable oversight

4. INTERPRETABILITY:
   - Understand what model is doing
   - Mechanistic interpretability
   - Feature visualization

EXISTENTIAL RISK:
=================

Concerns about advanced AI:
    - Misaligned superintelligence
    - Loss of human control
    - Concentration of power

Organizations working on this:
    - Anthropic
    - OpenAI
    - DeepMind
    - MIRI
    - Center for AI Safety

CURRENT SAFETY PRACTICES:
=========================

1. Red teaming: Adversarial testing
2. Safety evaluations: Benchmark dangerous capabilities
3. Staged deployment: Gradual rollout
4. Monitoring: Track for misuse
5. Incident response: Plan for failures
    """)


# =============================================================================
# SECTION 5: RESPONSIBLE AI PRACTICES
# =============================================================================

def responsible_ai_practices():
    """Responsible AI practices"""
    print("\n" + "=" * 70)
    print("RESPONSIBLE AI PRACTICES")
    print("=" * 70)
    
    print("""
RESPONSIBLE AI FRAMEWORK:
=========================

1. ACCOUNTABILITY
   - Clear ownership of AI systems
   - Defined roles and responsibilities
   - Audit trails

2. TRANSPARENCY
   - Document model decisions
   - Explain to affected parties
   - Open about limitations

3. FAIRNESS
   - Test for bias
   - Monitor for discrimination
   - Remediate issues

4. PRIVACY
   - Minimize data collection
   - Protect personal information
   - Respect consent

5. SAFETY
   - Test thoroughly
   - Plan for failures
   - Human oversight

6. HUMAN CONTROL
   - Meaningful human oversight
   - Ability to override
   - Clear escalation paths

IMPLEMENTATION CHECKLIST:
=========================

BEFORE DEVELOPMENT:
[ ] Define intended use and users
[ ] Identify potential harms
[ ] Assess regulatory requirements
[ ] Plan for diverse team

DURING DEVELOPMENT:
[ ] Document data sources
[ ] Test for bias
[ ] Implement privacy protections
[ ] Create model cards

BEFORE DEPLOYMENT:
[ ] Conduct impact assessment
[ ] Red team testing
[ ] Plan monitoring
[ ] Prepare incident response

AFTER DEPLOYMENT:
[ ] Monitor for drift
[ ] Track for misuse
[ ] Collect feedback
[ ] Regular audits

MODEL CARDS:
============

Document for each model:
    - Intended use
    - Training data
    - Performance metrics
    - Limitations
    - Ethical considerations
    - Bias evaluation results
    """)


# =============================================================================
# SECTION 6: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common AI ethics interview questions"""
    print("\n" + "=" * 70)
    print("AI ETHICS INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is algorithmic bias and how do you detect it?",
         "Systematic errors that create unfair outcomes. Detect by: disaggregated metrics across groups, fairness audits, testing with diverse data, comparing outcomes across protected attributes."),
        ("How do you balance accuracy and fairness?",
         "Often trade-off exists. Approach: Define acceptable fairness constraints, optimize accuracy within constraints, involve stakeholders in decisions, document trade-offs."),
        ("What is the alignment problem?",
         "Ensuring AI systems pursue goals beneficial to humans. Challenges: specifying values, maintaining control, robustness. Solutions: RLHF, constitutional AI, interpretability."),
        ("How do you handle privacy in ML?",
         "Techniques: differential privacy, federated learning, data minimization, anonymization. Also: consent, access controls, retention policies, compliance with regulations."),
        ("What would you do if you discovered bias in a deployed model?",
         "Assess severity, document issue, notify stakeholders, implement short-term mitigation, investigate root cause, develop long-term fix, update monitoring."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 7: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common AI ethics mistakes"""
    print("\n" + "=" * 70)
    print("COMMON AI ETHICS PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Ethics as afterthought", "Integrate ethics from project start, not end"),
        ("Homogeneous teams", "Diverse teams catch more issues; include varied perspectives"),
        ("Ignoring context", "Same model may be ethical in one context, harmful in another"),
        ("Over-relying on metrics", "Fairness metrics don't capture everything; use qualitative assessment too"),
        ("Assuming neutrality", "All AI embeds values; be explicit about choices"),
        ("No monitoring", "Bias can emerge over time; continuous monitoring essential"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 15: AI ETHICS & SAFETY - COMPREHENSIVE COVERAGE")
    print("=" * 70)
    
    assistant = EthicsAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is AI Ethics? (4W+H)")
        print("  2. Bias and Fairness")
        print("  3. Privacy and Security")
        print("  4. AI Safety and Alignment")
        print("  5. Responsible AI Practices")
        print()
        print("DEEP DIVE:")
        print("  6. Interview Questions")
        print("  7. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  8. AI Explanation (OpenAI)")
        print("  9. AI Explanation (Ollama - FREE)")
        print()
        print("  10. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-10): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_ethics_comprehensive()
        elif choice == "2":
            bias_and_fairness()
        elif choice == "3":
            privacy_and_security()
        elif choice == "4":
            ai_safety_alignment()
        elif choice == "5":
            responsible_ai_practices()
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
            explain_ethics_comprehensive()
            bias_and_fairness()
            privacy_and_security()
            ai_safety_alignment()
            responsible_ai_practices()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
