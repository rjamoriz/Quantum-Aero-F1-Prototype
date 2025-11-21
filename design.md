# System Design: Quantum-Aero F1 Prototype

## Architecture Overview

The system follows a microservice architecture to allow independent scaling of physics, ML, and quantum components.

```
+---------------------+        +------------------+        +-----------------------+
|   React Front-End   | <----> |   Node Backend   | <----> |  MongoDB Data Layer   |
+---------------------+        +------------------+        +-----------------------+
               |                          |                         |
               v                          v                         v
        +-------------+     +------------------------+     +----------------------+
        | Physics API |     | ML Inference Service   |     | Quantum Optimizer    |
        |  (VLM/Pan)  |     |  (PyTorch/ONNX CUDA)   |     | (QAOA / QUBO Model)  |
        +-------------+     +------------------------+     +----------------------+
```

## Components

### 1. Front-End (Next.js + React)

* Dark-mode UI.
* Three.js and VTK.js for aerodynamic field visualization.
* Control panel for simulation and optimization parameters.
* Real-time charts (drag, Cl, Cd, pressure).

### 2. Backend (Node.js + Express)

* Central hub for orchestrating calls to microservices.
* Manages simulation jobs and model states.
* REST/GraphQL hybrid interface.
* JWT authentication for multi-user scenarios.

### 3. ML Surrogate Service

* PyTorch CUDA surrogate model.
* ONNX export + ONNX Runtime GPU for inference.
* Models: pressure map predictor, drag/downforce predictor, vortex intensity predictor.
* Auto-scheduler for batching requests.

### 4. Physics Microservice

* Aerodynamic computation using VLM + Panel methods.
* CPU implementation + optional CUDA acceleration.
* Mesh loader for STL/OBJ.
* Validates ML surrogate model.

### 5. Quantum Optimization Service

* Defines aerodynamic objectives as QUBOs.
* QAOA + classical optimizers.
* Connects to Qiskit-Aer local simulator.
* Optional adapter for D-Wave Ocean SDK.

### 6. Data Layer

* MongoDB collections:

  * `simulations`
  * `designs`
  * `optimizer_runs`
  * `surrogate_models`

### 7. Deployment Design

* Docker containers for each microservice.
* docker-compose with GPU-enabled containers.
* Reverse-proxy (NGINX) for routing.
* Logging & monitoring via Prometheus + Grafana.

## Data Flow

1. User configures a shape → uploaded mesh.
2. Front-end sends request to backend.
3. Backend triggers physics service + ML surrogate.
4. ML predicts fast aerodynamic fields.
5. Quantum service selects optimal structural parameters.
6. Backend stores results.
7. UI displays 3D fields + KPIs.

## Security & Compliance

* All services isolated in Docker network.
* JWT + HTTPS.
* Resource caps for GPU jobs.

## Performance Targets

* <100ms inference latency on RTX 4070.
* ≤2s full simulation pass.
* Quantum optimizer: <10 iterations per job.
