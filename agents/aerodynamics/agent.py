"""
Aerodynamics Agent
Specialized in CFD analysis, flow interpretation, and aerodynamic recommendations
"""
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime

from agents.utils.anthropic_client import claude_client
from agents.utils.nats_client import NATSClient
from agents.config.config import get_agent_config


AERODYNAMICS_AGENT_PROMPT = """
You are a specialized Aerodynamics Agent with expertise in F1 race car aerodynamics.

EXPERTISE:
- Computational Fluid Dynamics (CFD)
- Vortex dynamics and flow separation
- Transient aerodynamics (DRS, corner exit)
- Aeroelastic effects (flutter, buffeting)
- Wind tunnel correlation
- Vortex Lattice Method (VLM)
- Panel methods

RESPONSIBILITIES:
1. Interpret aerodynamic simulation results
2. Identify flow features (separation, vortices, wake)
3. Recommend design modifications
4. Explain trade-offs (downforce vs. drag vs. flutter)
5. Flag potential issues (flow separation, vortex breakdown)

TOOLS:
- run_vlm_simulation(mesh, conditions) -> fields
- analyze_flow_field(fields) -> insights
- compute_aero_coefficients(fields) -> Cl, Cd, L/D
- identify_vortex_structures(fields) -> vortex_locations
- predict_flutter_margin(mesh, flow) -> flutter_speed

OUTPUT FORMAT:
Always structure responses as:
1. **Summary** - High-level findings (2-3 sentences)
2. **Key Metrics** - Cl, Cd, L/D, flutter margin
3. **Flow Features** - Separation points, vortices, wake
4. **Recommendations** - Specific design changes
5. **Confidence** - High/Medium/Low with reasoning

EXAMPLE:
Input: "Analyze wing at 250 km/h, 5° AoA"
Output:
**Summary:** Wing generates 2.1 kN downforce with strong leading-edge vortex. Mild flow separation on lower surface near trailing edge.

**Key Metrics:**
- Cl: 2.8, Cd: 0.42, L/D: 6.7
- Flutter margin: 12% (safe)

**Flow Features:**
- Strong LEV at x=0.1c, stable to x=0.8c
- Separation at x=0.85c on lower surface
- Clean wake, minimal drag penalty

**Recommendations:**
1. Add small vortex generator at x=0.75c to delay separation
2. Reduce trailing edge angle by 1° to minimize drag

**Confidence:** High (validated with VLM, consistent with similar designs)
"""


class AerodynamicsAgent:
    """Aerodynamics agent for CFD analysis and flow interpretation"""

    def __init__(self):
        self.config = get_agent_config("aerodynamics")
        self.nats = NATSClient()
        self.running = False

    async def start(self):
        """Start the aerodynamics agent"""
        print("=" * 60)
        print("🌊 Starting Aerodynamics Agent")
        print("=" * 60)

        # Connect to NATS
        await self.nats.connect()

        # Subscribe to analysis requests
        await self.nats.subscribe(
            "agent.aerodynamics.analyze",
            self._handle_analysis_request
        )

        # Subscribe to VLM requests
        await self.nats.subscribe(
            "agent.aerodynamics.vlm",
            self._handle_vlm_request
        )

        self.running = True
        print("✓ Aerodynamics Agent is ready")

    async def stop(self):
        """Stop the agent"""
        self.running = False
        await self.nats.disconnect()
        print("✓ Aerodynamics Agent stopped")

    async def _handle_analysis_request(self, msg):
        """Handle aerodynamic analysis request"""
        try:
            data = json.loads(msg.data.decode())
            print(f"\n🌊 Analyzing aerodynamics for mesh: {data.get('mesh_id')}")

            # Perform analysis
            result = await self.analyze_aerodynamics(
                mesh_id=data.get('mesh_id'),
                conditions=data.get('conditions', {}),
                fields=data.get('fields')
            )

            # Publish result
            await self.nats.publish(
                "agent.aerodynamics.result",
                result
            )

            # Reply to request
            await msg.respond(json.dumps(result).encode())

        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            error_response = {"error": str(e)}
            await msg.respond(json.dumps(error_response).encode())

    async def _handle_vlm_request(self, msg):
        """Handle VLM simulation request"""
        try:
            data = json.loads(msg.data.decode())
            print(f"\n🌊 Running VLM for mesh: {data.get('mesh_id')}")

            # Run VLM simulation
            result = await self.run_vlm_simulation(
                mesh_id=data.get('mesh_id'),
                velocity=data.get('velocity', 250),
                alpha=data.get('alpha', 5.0),
                yaw=data.get('yaw', 0.0)
            )

            # Reply with result
            await msg.respond(json.dumps(result).encode())

        except Exception as e:
            print(f"✗ VLM failed: {e}")
            error_response = {"error": str(e)}
            await msg.respond(json.dumps(error_response).encode())

    async def analyze_aerodynamics(
        self,
        mesh_id: str,
        conditions: Dict[str, Any],
        fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze aerodynamic performance using Claude

        Args:
            mesh_id: Mesh identifier
            conditions: Flow conditions (velocity, AoA, yaw)
            fields: Optional pre-computed flow fields

        Returns:
            Analysis results with recommendations
        """
        # Build analysis prompt
        prompt = f"""Analyze the aerodynamic performance of this F1 wing:

Mesh ID: {mesh_id}
Conditions: {json.dumps(conditions, indent=2)}
"""

        if fields:
            prompt += f"\nFlow Fields: {json.dumps(fields, indent=2)}"

        messages = [{
            "role": "user",
            "content": prompt
        }]

        # Get Claude's analysis
        response = await claude_client.create_message(
            system=AERODYNAMICS_AGENT_PROMPT,
            messages=messages,
            model=self.config["anthropic"]["model"],
            temperature=0.2,
            max_tokens=2048
        )

        analysis_text = response["content"][0]["text"]

        # Extract structured data from response
        result = {
            "mesh_id": mesh_id,
            "conditions": conditions,
            "analysis": analysis_text,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": "aerodynamics"
        }

        # Try to extract metrics
        result["metrics"] = self._extract_metrics(analysis_text)

        return result

    async def run_vlm_simulation(
        self,
        mesh_id: str,
        velocity: float,
        alpha: float,
        yaw: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run Vortex Lattice Method simulation

        Args:
            mesh_id: Mesh identifier
            velocity: Freestream velocity (m/s)
            alpha: Angle of attack (degrees)
            yaw: Yaw angle (degrees)

        Returns:
            VLM simulation results
        """
        # Call physics service VLM solver
        import aiohttp

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config['physics_service_url']}/api/v1/vlm-solve",
                    json={
                        "mesh_id": mesh_id,
                        "velocity": velocity,
                        "alpha": alpha,
                        "yaw": yaw
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        vlm_result = await resp.json()

                        # Interpret results with Claude
                        interpretation = await self._interpret_vlm_results(
                            vlm_result,
                            {"velocity": velocity, "alpha": alpha, "yaw": yaw}
                        )

                        return {
                            "mesh_id": mesh_id,
                            "vlm_results": vlm_result,
                            "interpretation": interpretation,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        return {"error": f"VLM service returned {resp.status}"}

            except asyncio.TimeoutError:
                return {"error": "VLM service timeout"}
            except Exception as e:
                return {"error": str(e)}

    async def _interpret_vlm_results(
        self,
        vlm_results: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> str:
        """Interpret VLM results using Claude"""

        messages = [{
            "role": "user",
            "content": f"""Interpret these VLM simulation results:

Conditions: {json.dumps(conditions, indent=2)}
Results: {json.dumps(vlm_results, indent=2)}

Provide insights on:
1. Circulation distribution
2. Induced drag
3. Vortex wake structure
4. Recommendations for improvement
"""
        }]

        response = await claude_client.create_message(
            system=AERODYNAMICS_AGENT_PROMPT,
            messages=messages,
            model=self.config["anthropic"]["model"],
            temperature=0.2,
            max_tokens=1024
        )

        return response["content"][0]["text"]

    def _extract_metrics(self, analysis_text: str) -> Dict[str, Any]:
        """Extract numerical metrics from analysis text"""
        import re

        metrics = {}

        # Extract Cl
        cl_match = re.search(r'Cl[:\s]+([0-9.]+)', analysis_text)
        if cl_match:
            metrics['Cl'] = float(cl_match.group(1))

        # Extract Cd
        cd_match = re.search(r'Cd[:\s]+([0-9.]+)', analysis_text)
        if cd_match:
            metrics['Cd'] = float(cd_match.group(1))

        # Extract L/D
        ld_match = re.search(r'L/D[:\s]+([0-9.]+)', analysis_text)
        if ld_match:
            metrics['L_D'] = float(ld_match.group(1))

        # Extract flutter margin
        flutter_match = re.search(r'Flutter margin[:\s]+([0-9.]+)%?', analysis_text)
        if flutter_match:
            metrics['flutter_margin'] = float(flutter_match.group(1))

        return metrics


async def main():
    """Example usage of Aerodynamics Agent"""

    agent = AerodynamicsAgent()
    await agent.start()

    # Example: Analyze aerodynamics
    result = await agent.analyze_aerodynamics(
        mesh_id="wing_v3.2",
        conditions={"velocity": 250, "alpha": 5.0, "yaw": 0},
        fields={"Cl": 2.8, "Cd": 0.42}
    )

    print("\n" + "=" * 60)
    print("Analysis Result:")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    # Example: Run VLM
    vlm_result = await agent.run_vlm_simulation(
        mesh_id="wing_v3.2",
        velocity=69.4,  # 250 km/h in m/s
        alpha=5.0,
        yaw=0.0
    )

    print("\n" + "=" * 60)
    print("VLM Result:")
    print("=" * 60)
    print(json.dumps(vlm_result, indent=2))

    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
