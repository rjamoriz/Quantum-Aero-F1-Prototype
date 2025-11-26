"""
F1 Telemetry Loop
Real-time track data integration with Kafka + TimescaleDB
Target: <1s optimization latency
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import time
import json
from datetime import datetime
from collections import deque


@dataclass
class TelemetryData:
    """Real-time telemetry from F1 car"""
    timestamp: float
    lap_number: int
    position: float  # Track position (0-1)
    velocity: float  # m/s
    throttle: float  # 0-1
    brake: float  # 0-1
    steering: float  # -1 to 1
    drs_active: bool
    front_wing_angle: float  # degrees
    rear_wing_angle: float  # degrees
    downforce: float  # N
    drag: float  # N
    tire_temps: List[float]  # [FL, FR, RL, RR] °C
    track_temp: float  # °C
    air_temp: float  # °C
    wind_speed: float  # m/s
    wind_direction: float  # degrees


@dataclass
class OptimizationCommand:
    """Optimization command to car"""
    timestamp: float
    drs_enable: bool
    front_wing_delta: float  # degrees
    rear_wing_delta: float  # degrees
    confidence: float
    lap_time_improvement: float  # seconds


class KafkaProducer:
    """Mock Kafka producer for telemetry streaming"""
    
    def __init__(self, topic: str = "f1_telemetry"):
        self.topic = topic
        self.message_count = 0
        print(f"Kafka Producer initialized: topic={topic}")
    
    def send(self, message: Dict[str, Any]):
        """Send message to Kafka topic"""
        self.message_count += 1
        # In production: kafka.KafkaProducer().send(self.topic, message)
    
    def flush(self):
        """Flush pending messages"""
        pass


class TimescaleDBClient:
    """Mock TimescaleDB client for time-series storage"""
    
    def __init__(self, connection_string: str = "postgresql://localhost:5432/f1_telemetry"):
        self.connection_string = connection_string
        self.data_buffer = []
        print(f"TimescaleDB Client initialized")
    
    def insert_telemetry(self, data: TelemetryData):
        """Insert telemetry data"""
        self.data_buffer.append(asdict(data))
        # In production: INSERT INTO telemetry VALUES (...)
    
    def query_lap_data(self, lap_number: int) -> List[Dict[str, Any]]:
        """Query telemetry for specific lap"""
        # In production: SELECT * FROM telemetry WHERE lap_number = ?
        return [d for d in self.data_buffer if d.get('lap_number') == lap_number]
    
    def get_historical_average(self, track_position: float, window: int = 10) -> Dict[str, float]:
        """Get historical average at track position"""
        # Simplified mock
        return {
            'avg_velocity': 75.0 + np.random.randn() * 5,
            'avg_downforce': 5000.0 + np.random.randn() * 200,
            'avg_drag': 800.0 + np.random.randn() * 50
        }


class F1TelemetryLoop:
    """
    Real-time F1 telemetry processing and optimization
    
    Features:
    - Kafka streaming (1000+ msg/s)
    - TimescaleDB storage
    - Real-time optimization (<1s)
    - Race strategy engine
    - Historical analysis
    """
    
    def __init__(
        self,
        kafka_topic: str = "f1_telemetry",
        timescale_conn: str = "postgresql://localhost:5432/f1_telemetry",
        optimization_interval: float = 1.0  # seconds
    ):
        self.kafka_producer = KafkaProducer(kafka_topic)
        self.timescale_db = TimescaleDBClient(timescale_conn)
        self.optimization_interval = optimization_interval
        
        # Real-time buffers
        self.telemetry_buffer = deque(maxlen=1000)
        self.optimization_history = []
        
        # Performance tracking
        self.processing_times = []
        self.optimization_times = []
        
        # Current state
        self.current_lap = 0
        self.last_optimization = 0
        
        print(f"F1 Telemetry Loop initialized")
        print(f"  Optimization interval: {optimization_interval}s")
    
    def ingest_telemetry(self, data: TelemetryData):
        """
        Ingest real-time telemetry data
        
        Target: 1000+ messages/second
        """
        start_time = time.time()
        
        # Add to buffer
        self.telemetry_buffer.append(data)
        
        # Stream to Kafka
        self.kafka_producer.send(asdict(data))
        
        # Store in TimescaleDB
        self.timescale_db.insert_telemetry(data)
        
        # Update current lap
        if data.lap_number > self.current_lap:
            self.current_lap = data.lap_number
            print(f"Lap {self.current_lap} started")
        
        # Track processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        # Keep last 1000 measurements
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)
    
    def optimize_realtime(self, current_data: TelemetryData) -> Optional[OptimizationCommand]:
        """
        Real-time aerodynamic optimization
        
        Target: <1 second latency
        """
        # Check if optimization needed
        if time.time() - self.last_optimization < self.optimization_interval:
            return None
        
        start_time = time.time()
        
        # Get historical data for this track position
        historical = self.timescale_db.get_historical_average(current_data.position)
        
        # Current performance
        current_velocity = current_data.velocity
        current_downforce = current_data.downforce
        current_drag = current_data.drag
        
        # Optimization logic
        # 1. Compare with historical average
        velocity_delta = current_velocity - historical['avg_velocity']
        
        # 2. Determine optimal wing angles
        # If slower than average, reduce drag (reduce wing angles)
        # If faster, maintain or increase downforce for corners
        
        if velocity_delta < -2.0:  # Significantly slower
            # Reduce drag
            front_wing_delta = -1.0
            rear_wing_delta = -1.5
            drs_enable = current_data.position < 0.3 or current_data.position > 0.7  # Straights
        elif velocity_delta > 2.0:  # Significantly faster
            # Maintain current setup
            front_wing_delta = 0.0
            rear_wing_delta = 0.0
            drs_enable = False
        else:
            # Fine-tune based on track section
            if 0.2 < current_data.position < 0.4:  # Corner section
                front_wing_delta = 0.5
                rear_wing_delta = 0.5
                drs_enable = False
            else:  # Straight section
                front_wing_delta = -0.5
                rear_wing_delta = -0.5
                drs_enable = True
        
        # Estimate lap time improvement (simplified)
        drag_reduction = abs(front_wing_delta + rear_wing_delta) * 10  # N
        velocity_gain = drag_reduction / 100  # m/s (simplified)
        lap_time_improvement = velocity_gain * 0.1  # seconds
        
        # Create command
        command = OptimizationCommand(
            timestamp=time.time(),
            drs_enable=drs_enable,
            front_wing_delta=front_wing_delta,
            rear_wing_delta=rear_wing_delta,
            confidence=0.85 + np.random.rand() * 0.1,
            lap_time_improvement=lap_time_improvement
        )
        
        # Track optimization time
        optimization_time = (time.time() - start_time) * 1000  # ms
        self.optimization_times.append(optimization_time)
        
        if len(self.optimization_times) > 100:
            self.optimization_times.pop(0)
        
        # Update last optimization time
        self.last_optimization = time.time()
        
        # Store in history
        self.optimization_history.append(command)
        
        # Stream to Kafka
        self.kafka_producer.send({
            'type': 'optimization_command',
            **asdict(command)
        })
        
        return command
    
    def analyze_lap(self, lap_number: int) -> Dict[str, Any]:
        """
        Analyze completed lap performance
        
        Returns:
            Lap analysis with recommendations
        """
        # Get lap data from database
        lap_data = self.timescale_db.query_lap_data(lap_number)
        
        if not lap_data:
            return {'status': 'no_data'}
        
        # Extract metrics
        velocities = [d['velocity'] for d in lap_data]
        downforces = [d['downforce'] for d in lap_data]
        drags = [d['drag'] for d in lap_data]
        
        # Compute statistics
        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        avg_downforce = np.mean(downforces)
        avg_drag = np.mean(drags)
        
        # Estimate lap time (simplified)
        track_length = 5000  # meters
        lap_time = track_length / avg_velocity
        
        # Find optimization opportunities
        # Sections where velocity was below average
        slow_sections = []
        for i, d in enumerate(lap_data):
            if d['velocity'] < avg_velocity - 5:
                slow_sections.append({
                    'position': d['position'],
                    'velocity': d['velocity'],
                    'deficit': avg_velocity - d['velocity']
                })
        
        return {
            'lap_number': lap_number,
            'lap_time': lap_time,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'avg_downforce': avg_downforce,
            'avg_drag': avg_drag,
            'l_d_ratio': avg_downforce / avg_drag if avg_drag > 0 else 0,
            'num_data_points': len(lap_data),
            'slow_sections': slow_sections[:5],  # Top 5
            'recommendations': self._generate_recommendations(lap_data)
        }
    
    def _generate_recommendations(self, lap_data: List[Dict[str, Any]]) -> List[str]:
        """Generate setup recommendations based on lap data"""
        recommendations = []
        
        # Analyze DRS usage
        drs_active_count = sum(1 for d in lap_data if d.get('drs_active', False))
        drs_usage_pct = (drs_active_count / len(lap_data)) * 100
        
        if drs_usage_pct < 10:
            recommendations.append("Increase DRS usage on straights for higher top speed")
        
        # Analyze wing angles
        avg_front_wing = np.mean([d['front_wing_angle'] for d in lap_data])
        avg_rear_wing = np.mean([d['rear_wing_angle'] for d in lap_data])
        
        if avg_front_wing > 10:
            recommendations.append("Consider reducing front wing angle to decrease drag")
        
        if avg_rear_wing < -5:
            recommendations.append("Increase rear wing angle for better corner stability")
        
        # Analyze tire temperatures
        avg_tire_temp = np.mean([np.mean(d['tire_temps']) for d in lap_data])
        
        if avg_tire_temp > 100:
            recommendations.append("Tire temperatures high - consider cooling adjustments")
        elif avg_tire_temp < 80:
            recommendations.append("Tire temperatures low - increase downforce for grip")
        
        return recommendations
    
    def get_race_strategy(self, remaining_laps: int) -> Dict[str, Any]:
        """
        Generate race strategy recommendations
        
        Args:
            remaining_laps: Number of laps remaining
        
        Returns:
            Strategy recommendations
        """
        # Analyze recent performance
        recent_optimizations = self.optimization_history[-10:]
        
        if not recent_optimizations:
            return {'status': 'insufficient_data'}
        
        # Average lap time improvement
        avg_improvement = np.mean([opt.lap_time_improvement for opt in recent_optimizations])
        total_potential_gain = avg_improvement * remaining_laps
        
        # Tire degradation estimate (simplified)
        tire_deg_per_lap = 0.05  # seconds
        tire_penalty = tire_deg_per_lap * remaining_laps
        
        # Net gain
        net_gain = total_potential_gain - tire_penalty
        
        return {
            'remaining_laps': remaining_laps,
            'avg_lap_improvement': avg_improvement,
            'total_potential_gain': total_potential_gain,
            'tire_degradation_penalty': tire_penalty,
            'net_time_gain': net_gain,
            'strategy': 'aggressive' if net_gain > 2.0 else 'conservative',
            'pit_stop_recommended': tire_penalty > 5.0,
            'confidence': 0.8
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get telemetry loop performance metrics"""
        if not self.processing_times:
            return {'status': 'no_data'}
        
        return {
            'total_messages': len(self.telemetry_buffer),
            'avg_processing_time_ms': float(np.mean(self.processing_times)),
            'max_processing_time_ms': float(np.max(self.processing_times)),
            'messages_per_second': 1000.0 / np.mean(self.processing_times) if self.processing_times else 0,
            'avg_optimization_time_ms': float(np.mean(self.optimization_times)) if self.optimization_times else 0,
            'max_optimization_time_ms': float(np.max(self.optimization_times)) if self.optimization_times else 0,
            'target_optimization_ms': 1000,
            'target_met': all(t < 1000 for t in self.optimization_times) if self.optimization_times else False,
            'total_optimizations': len(self.optimization_history),
            'current_lap': self.current_lap
        }


if __name__ == "__main__":
    # Test telemetry loop
    print("Testing F1 Telemetry Loop\n")
    
    loop = F1TelemetryLoop(optimization_interval=0.5)
    
    # Simulate race
    print("Simulating race data...")
    
    for lap in range(1, 4):
        print(f"\n--- Lap {lap} ---")
        
        # Simulate 100 telemetry points per lap
        for i in range(100):
            position = i / 100.0
            
            # Generate synthetic telemetry
            telemetry = TelemetryData(
                timestamp=time.time(),
                lap_number=lap,
                position=position,
                velocity=70 + np.sin(position * 2 * np.pi) * 20 + np.random.randn() * 2,
                throttle=0.8 + np.random.rand() * 0.2,
                brake=0.2 if 0.3 < position < 0.5 else 0.0,
                steering=np.sin(position * 4 * np.pi) * 0.5,
                drs_active=position < 0.3 or position > 0.7,
                front_wing_angle=5.0 + np.random.randn(),
                rear_wing_angle=8.0 + np.random.randn(),
                downforce=5000 + np.random.randn() * 200,
                drag=800 + np.random.randn() * 50,
                tire_temps=[90 + np.random.randn() * 5 for _ in range(4)],
                track_temp=30.0,
                air_temp=25.0,
                wind_speed=2.0,
                wind_direction=45.0
            )
            
            # Ingest telemetry
            loop.ingest_telemetry(telemetry)
            
            # Try optimization
            command = loop.optimize_realtime(telemetry)
            if command:
                print(f"  Optimization at position {position:.2f}: "
                      f"DRS={'ON' if command.drs_enable else 'OFF'}, "
                      f"Wing Δ={command.front_wing_delta:.1f}°/{command.rear_wing_delta:.1f}°, "
                      f"Gain={command.lap_time_improvement:.3f}s")
            
            time.sleep(0.001)  # Simulate real-time
        
        # Analyze lap
        analysis = loop.analyze_lap(lap)
        print(f"\nLap {lap} Analysis:")
        print(f"  Lap time: {analysis['lap_time']:.2f}s")
        print(f"  Avg velocity: {analysis['avg_velocity']:.1f} m/s")
        print(f"  L/D ratio: {analysis['l_d_ratio']:.2f}")
        print(f"  Recommendations: {len(analysis['recommendations'])}")
        for rec in analysis['recommendations']:
            print(f"    - {rec}")
    
    # Get race strategy
    print("\n--- Race Strategy ---")
    strategy = loop.get_race_strategy(remaining_laps=20)
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Potential gain: {strategy['total_potential_gain']:.2f}s")
    print(f"  Net gain: {strategy['net_time_gain']:.2f}s")
    print(f"  Pit stop: {'Yes' if strategy['pit_stop_recommended'] else 'No'}")
    
    # Get performance metrics
    print("\n--- Performance Metrics ---")
    metrics = loop.get_performance_metrics()
    print(f"  Messages processed: {metrics['total_messages']}")
    print(f"  Avg processing: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"  Messages/sec: {metrics['messages_per_second']:.0f}")
    print(f"  Avg optimization: {metrics['avg_optimization_time_ms']:.2f}ms")
    print(f"  Target met: {'✓' if metrics['target_met'] else '✗'}")
    
    print("\n✓ Telemetry loop test complete!")
