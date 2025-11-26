"""
F1 Aerodynamic Control Environment
Gym environment for RL-based DRS and flap optimization
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, Tuple, Optional
import time


class F1AeroControlEnv(gym.Env):
    """
    Reinforcement Learning environment for F1 aerodynamic control
    
    State: [velocity, position, track_curvature, competitors, drs_state, flap_angles]
    Action: [drs_activation, front_flap_angle, rear_flap_angle]
    Reward: Lap time improvement + safety constraints
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        track_length: float = 5000.0,  # meters
        max_velocity: float = 90.0,  # m/s (324 km/h)
        use_cfd_surrogate: bool = True
    ):
        super(F1AeroControlEnv, self).__init__()
        
        self.track_length = track_length
        self.max_velocity = max_velocity
        self.use_cfd_surrogate = use_cfd_surrogate
        
        # Track segments (simplified)
        self.num_segments = 20
        self.segment_length = track_length / self.num_segments
        
        # Generate track profile
        self.track_curvature = self._generate_track()
        
        # State space
        # [velocity, position, segment_curvature, next_curvature, drs_available, 
        #  front_flap, rear_flap, downforce, drag, lap_time]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, -15, -15, 0, 0, 0]),
            high=np.array([100, 1, 1, 1, 1, 15, 15, 5, 2, 200]),
            dtype=np.float32
        )
        
        # Action space
        # [drs_activation (0/1), front_flap_delta, rear_flap_delta]
        self.action_space = spaces.Box(
            low=np.array([0, -5, -5]),
            high=np.array([1, 5, 5]),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        print("F1 Aero Control Environment initialized")
        print(f"  Track length: {track_length}m")
        print(f"  Segments: {self.num_segments}")
        print(f"  Max velocity: {max_velocity} m/s")
    
    def _generate_track(self) -> np.ndarray:
        """Generate realistic F1 track profile"""
        curvature = np.zeros(self.num_segments)
        
        # Add straights and corners
        for i in range(self.num_segments):
            if i % 5 == 0:  # Corner
                curvature[i] = 0.5 + np.random.rand() * 0.5
            elif i % 5 == 1:  # Exit
                curvature[i] = 0.3
            elif i % 5 == 2:  # Straight
                curvature[i] = 0.0
            else:  # Approach
                curvature[i] = 0.2
        
        return curvature
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.position = 0.0
        self.velocity = 50.0  # Start at moderate speed
        self.lap_time = 0.0
        self.timestep = 0
        
        # Flap angles (degrees)
        self.front_flap = 0.0
        self.rear_flap = 0.0
        
        # DRS state
        self.drs_active = False
        self.drs_available = True
        
        # Aerodynamic coefficients
        self.cl = 2.5
        self.cd = 0.45
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Current segment
        segment = int((self.position / self.track_length) * self.num_segments) % self.num_segments
        next_segment = (segment + 1) % self.num_segments
        
        # Normalize position
        norm_position = (self.position % self.track_length) / self.track_length
        
        obs = np.array([
            self.velocity / self.max_velocity,  # Normalized velocity
            norm_position,  # Normalized position
            self.track_curvature[segment],  # Current curvature
            self.track_curvature[next_segment],  # Next curvature
            float(self.drs_available),  # DRS available
            self.front_flap / 15.0,  # Normalized front flap
            self.rear_flap / 15.0,  # Normalized rear flap
            self.cl / 3.0,  # Normalized downforce
            self.cd / 1.0,  # Normalized drag
            self.lap_time / 100.0  # Normalized lap time
        ], dtype=np.float32)
        
        return obs
    
    def _compute_aerodynamics(self) -> Tuple[float, float]:
        """
        Compute aerodynamic coefficients based on flap angles and DRS
        
        Returns:
            (cl, cd) - Lift and drag coefficients
        """
        # Base coefficients
        cl_base = 2.5
        cd_base = 0.45
        
        # Flap effects
        cl_front = 0.02 * self.front_flap
        cl_rear = 0.03 * self.rear_flap
        
        cd_front = 0.005 * abs(self.front_flap)
        cd_rear = 0.008 * abs(self.rear_flap)
        
        # DRS effect (reduce drag, reduce downforce)
        if self.drs_active:
            cl_drs = -0.3
            cd_drs = -0.15
        else:
            cl_drs = 0.0
            cd_drs = 0.0
        
        # Total coefficients
        cl = cl_base + cl_front + cl_rear + cl_drs
        cd = cd_base + cd_front + cd_rear + cd_drs
        
        # Clamp to physical limits
        cl = np.clip(cl, 0.5, 3.5)
        cd = np.clip(cd, 0.2, 0.8)
        
        return cl, cd
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one timestep
        
        Args:
            action: [drs_activation, front_flap_delta, rear_flap_delta]
        
        Returns:
            (observation, reward, done, info)
        """
        dt = 0.1  # 100ms timestep
        
        # Parse action
        drs_cmd = action[0] > 0.5
        front_flap_delta = np.clip(action[1], -5, 5)
        rear_flap_delta = np.clip(action[2], -5, 5)
        
        # Update flap angles
        self.front_flap = np.clip(self.front_flap + front_flap_delta, -15, 15)
        self.rear_flap = np.clip(self.rear_flap + rear_flap_delta, -15, 15)
        
        # Update DRS (only if available and straight)
        segment = int((self.position / self.track_length) * self.num_segments) % self.num_segments
        is_straight = self.track_curvature[segment] < 0.1
        
        if drs_cmd and self.drs_available and is_straight:
            self.drs_active = True
        else:
            self.drs_active = False
        
        # Compute aerodynamics
        self.cl, self.cd = self._compute_aerodynamics()
        
        # Physics simulation
        # Simplified: F = ma, drag force, downforce
        mass = 798.0  # kg (F1 minimum weight)
        air_density = 1.225  # kg/m³
        frontal_area = 1.5  # m²
        
        # Aerodynamic forces
        drag_force = 0.5 * air_density * frontal_area * self.cd * self.velocity**2
        downforce = 0.5 * air_density * frontal_area * self.cl * self.velocity**2
        
        # Engine power (simplified)
        max_power = 1000000  # 1000 kW
        power = max_power * (1.0 - self.velocity / self.max_velocity)
        
        # Acceleration
        accel = (power / self.velocity - drag_force) / mass if self.velocity > 0 else 0
        
        # Corner braking (simplified)
        if self.track_curvature[segment] > 0.3:
            # Need more downforce in corners
            if self.cl < 2.0:
                accel -= 5.0  # Penalty for insufficient downforce
        
        # Update velocity
        self.velocity = np.clip(self.velocity + accel * dt, 0, self.max_velocity)
        
        # Update position
        self.position += self.velocity * dt
        self.lap_time += dt
        self.timestep += 1
        
        # Compute reward
        reward = self._compute_reward(accel, segment)
        
        # Check if lap complete
        done = self.position >= self.track_length
        
        # Info
        info = {
            'lap_time': self.lap_time,
            'velocity': self.velocity,
            'cl': self.cl,
            'cd': self.cd,
            'drs_active': self.drs_active,
            'segment': segment,
            'position': self.position
        }
        
        return self._get_observation(), reward, done, info
    
    def _compute_reward(self, accel: float, segment: int) -> float:
        """
        Compute reward for current state
        
        Reward components:
        - Velocity (higher is better)
        - Lap time (lower is better)
        - Downforce in corners (safety)
        - Drag on straights (efficiency)
        """
        reward = 0.0
        
        # Velocity reward (encourage high speed)
        reward += self.velocity / self.max_velocity * 1.0
        
        # Corner handling (need downforce)
        if self.track_curvature[segment] > 0.3:
            if self.cl > 2.0:
                reward += 0.5  # Good downforce
            else:
                reward -= 1.0  # Insufficient downforce (safety penalty)
        
        # Straight efficiency (low drag)
        if self.track_curvature[segment] < 0.1:
            if self.cd < 0.4:
                reward += 0.3  # Low drag
            if self.drs_active:
                reward += 0.2  # DRS usage bonus
        
        # Lap time penalty (encourage faster laps)
        reward -= self.lap_time / 1000.0
        
        return reward
    
    def render(self, mode='human'):
        """Render environment (optional)"""
        if mode == 'human':
            segment = int((self.position / self.track_length) * self.num_segments) % self.num_segments
            print(f"Lap: {self.lap_time:.2f}s | "
                  f"Pos: {self.position:.0f}m | "
                  f"Vel: {self.velocity:.1f}m/s | "
                  f"Seg: {segment} | "
                  f"Cl: {self.cl:.2f} | "
                  f"Cd: {self.cd:.3f} | "
                  f"DRS: {'ON' if self.drs_active else 'OFF'}")


if __name__ == "__main__":
    # Test environment
    print("Testing F1 Aero Control Environment\n")
    
    env = F1AeroControlEnv()
    
    print("\nRunning random policy for 1 lap...")
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if env.timestep % 50 == 0:
            env.render()
    
    print(f"\nLap complete!")
    print(f"  Lap time: {info['lap_time']:.2f}s")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average velocity: {env.track_length / info['lap_time']:.1f} m/s")
    
    print("\n✓ Environment test complete!")
