"""
Master Orchestrator Agent
Coordinates all specialized agents and maintains conversation context
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from agents.utils.anthropic_client import claude_client
from agents.utils.nats_client import NATSClient
from agents.config.config import get_agent_config


MASTER_ORCHESTRATOR_PROMPT = """
You are the Master Orchestrator for the Quantum-Aero F1 aerodynamic optimization system.

ROLE:
- Coordinate all specialized agents to solve complex aerodynamic problems
- Maintain full conversation context with the user
- Decompose high-level requests into sub-tasks for specialized agents
- Synthesize results from multiple agents into coherent responses
- Detect and resolve conflicts between agent outputs
- Ensure engineering safety (flutter margins, stress limits, etc.)

CAPABILITIES:
You have access to 8 specialized agents:
1. Intent Router - Routes tasks to appropriate agents
2. Aerodynamics Agent - CFD simulations and analysis
3. ML Surrogate Agent - Fast ML predictions
4. Quantum Optimizer Agent - Design space optimization
5. Physics Validator Agent - Physics-based validation
6. Visualization Agent - Result visualization
7. Data Manager Agent - Historical data retrieval
8. Analysis Agent - Trade-off analysis
9. Report Generator Agent - Technical report generation

COMMUNICATION:
- Use NATS pub-sub for agent-to-agent messages
- Receive user queries in natural language
- Respond with technical accuracy and clear explanations
- Flag uncertainties and confidence levels
- Ask clarifying questions when needed

SAFETY:
- NEVER recommend designs that violate FIA regulations
- ALWAYS validate flutter margins (Vf > 1.2 × Vmax)
- CHECK structural stress limits before optimization
- WARN about potential safety issues

WORKFLOW:
When receiving a user query:
1. Analyze the request and identify required agents
2. Decompose into sub-tasks
3. Coordinate agent execution (parallel when possible)
4. Synthesize results
5. Respond to user with clear recommendations

EXAMPLES:
User: "Optimize this wing for Monza"
You: [Reasoning] Monza requires high top speed, so minimize drag while maintaining sufficient downforce for stability. I'll coordinate:
1. ML agent for fast design exploration
2. Quantum agent for optimization
3. Physics agent for validation
4. Analysis agent for trade-off recommendations

[Action] Coordinating agents...

OUTPUT FORMAT:
Always structure responses as:
1. **Understanding** - Paraphrase user request
2. **Approach** - Agents and workflow
3. **Results** - Key findings
4. **Recommendation** - Clear next steps
"""


class MasterOrchestratorAgent:
    """Master orchestrator agent that coordinates all specialized agents"""

    def __init__(self):
        self.config = get_agent_config("master_orchestrator")
        self.nats = NATSClient()
        self.conversation_history: List[Dict[str, str]] = []
        self.agent_results: Dict[str, Any] = {}
        self.running = False

    async def start(self):
        """Start the master orchestrator agent"""
        print("=" * 60)
        print("🚀 Starting Master Orchestrator Agent")
        print("=" * 60)

        # Connect to NATS
        await self.nats.connect()

        # Subscribe to relevant events
        await self.nats.subscribe_to_events(
            "analysis_ready",
            self._on_analysis_ready
        )

        self.running = True
        print("✓ Master Orchestrator Agent is ready")

    async def stop(self):
        """Stop the agent"""
        self.running = False
        await self.nats.disconnect()
        print("✓ Master Orchestrator Agent stopped")

    async def process_query(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Process user query with streaming response

        Args:
            user_query: User's natural language query
            context: Optional context (mesh_id, parameters, etc.)
            stream: Whether to stream the response

        Yields:
            Response chunks
        """
        print(f"\n📝 Processing query: {user_query[:50]}...")

        # Build context message
        context_str = json.dumps(context, indent=2) if context else "None"

        # Add to conversation history
        user_message = {
            "role": "user",
            "content": f"""User Query: {user_query}

Context: {context_str}

Current time: {datetime.utcnow().isoformat()}"""
        }

        messages = self.conversation_history + [user_message]

        # Stream Claude response
        full_response = ""

        if stream:
            async for chunk in claude_client.stream_message(
                system=MASTER_ORCHESTRATOR_PROMPT,
                messages=messages,
                model=self.config["anthropic"]["model"],
                temperature=self.config["anthropic"]["temperature"],
                max_tokens=self.config["anthropic"]["max_tokens"],
            ):
                full_response += chunk
                yield chunk
        else:
            response = await claude_client.create_message(
                system=MASTER_ORCHESTRATOR_PROMPT,
                messages=messages,
                model=self.config["anthropic"]["model"],
                temperature=self.config["anthropic"]["temperature"],
                max_tokens=self.config["anthropic"]["max_tokens"],
            )
            full_response = response["content"][0]["text"]
            yield full_response

        # Update conversation history
        self.conversation_history.append(user_message)
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

        # Keep conversation history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    async def decompose_task(
        self,
        user_query: str,
        mesh_id: str
    ) -> Dict[str, Any]:
        """
        Decompose user query into agent tasks

        Args:
            user_query: User's request
            mesh_id: Mesh identifier

        Returns:
            Task decomposition
        """
        messages = [{
            "role": "user",
            "content": f"""Decompose this aerodynamic optimization request into sub-tasks:

Query: {user_query}
Mesh ID: {mesh_id}

Return JSON with:
{{
  "ml_tasks": [
    {{"id": "variant_1", "parameters": {{"velocity": 250, "yaw": 0}}}},
    ...
  ],
  "needs_physics_validation": true/false,
  "needs_optimization": true/false,
  "objectives": ["maximize_downforce", "minimize_drag"],
  "constraints": ["flutter_Vf > 350 km/h", "mass < 5 kg"],
  "design_space": {{...}}
}}"""
        }]

        response = await claude_client.create_message(
            system=MASTER_ORCHESTRATOR_PROMPT,
            messages=messages,
            model=self.config["anthropic"]["model"],
            temperature=0.1,
            max_tokens=2048,
        )

        return await claude_client.extract_json_from_response(response)

    async def coordinate_agents(
        self,
        task_decomposition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents in parallel

        Args:
            task_decomposition: Task breakdown from decompose_task

        Returns:
            Aggregated results from all agents
        """
        tasks = []
        results = {}

        # ML predictions
        if task_decomposition.get("ml_tasks"):
            print(f"🧠 Requesting ML predictions for {len(task_decomposition['ml_tasks'])} variants")
            for task in task_decomposition["ml_tasks"]:
                tasks.append(self._call_ml_agent(task))

        # Physics validation
        if task_decomposition.get("needs_physics_validation"):
            print("⚛️  Requesting physics validation")
            tasks.append(self._call_physics_agent(task_decomposition))

        # Quantum optimization
        if task_decomposition.get("needs_optimization"):
            print("🔬 Requesting quantum optimization")
            tasks.append(self._call_quantum_agent(task_decomposition))

        # Execute all tasks in parallel
        if tasks:
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(agent_results):
                if isinstance(result, Exception):
                    print(f"✗ Agent task {i} failed: {result}")
                else:
                    results[f"task_{i}"] = result

        return results

    async def _call_ml_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Call ML Surrogate Agent"""
        try:
            response = await self.nats.request(
                "agent.ml.predict",
                task,
                timeout=10.0
            )
            return response
        except Exception as e:
            print(f"✗ ML agent call failed: {e}")
            return {"error": str(e)}

    async def _call_physics_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Call Physics Validator Agent"""
        try:
            response = await self.nats.request(
                "agent.physics.validate",
                task,
                timeout=30.0
            )
            return response
        except Exception as e:
            print(f"✗ Physics agent call failed: {e}")
            return {"error": str(e)}

    async def _call_quantum_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Call Quantum Optimizer Agent"""
        try:
            response = await self.nats.request(
                "agent.quantum.optimize",
                task,
                timeout=60.0
            )
            return response
        except Exception as e:
            print(f"✗ Quantum agent call failed: {e}")
            return {"error": str(e)}

    async def _on_analysis_ready(self, data: Dict[str, Any]):
        """Handle analysis ready event"""
        print(f"📊 Analysis ready: {data.get('analysis_id')}")
        self.agent_results["analysis"] = data


async def main():
    """Example usage of Master Orchestrator Agent"""

    # Initialize agent
    agent = MasterOrchestratorAgent()
    await agent.start()

    # Example 1: Process a query with streaming
    print("\n" + "=" * 60)
    print("Example 1: Streaming optimization query")
    print("=" * 60)

    user_query = "Optimize this wing for maximum downforce at Monza (250 km/h)"
    context = {
        "mesh_id": "wing_v3.2",
        "parameters": {"velocity": 250, "yaw": 0}
    }

    async for chunk in agent.process_query(user_query, context, stream=True):
        print(chunk, end="", flush=True)

    print("\n")

    # Example 2: Task decomposition
    print("\n" + "=" * 60)
    print("Example 2: Task decomposition")
    print("=" * 60)

    task_decomp = await agent.decompose_task(
        "Optimize for Monza high-speed track",
        "wing_v3.2"
    )
    print(json.dumps(task_decomp, indent=2))

    # Cleanup
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
