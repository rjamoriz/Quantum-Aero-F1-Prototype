# Project Plan: Quantum-Aero F1 Prototype

## Overview

This document defines the high-level planning for the Quantum‑Aero prototype: a hybrid Generative AI + Quantum Optimization + GPU‑accelerated aerodynamic simulator built for F1 aerodynamic research.

## Objectives

* Build a functional prototype to demonstrate combined AI + Quantum approaches for aerodynamic optimization.
* Run all components locally using NVIDIA RTX GPU and Docker.
* Deliver a professional‑grade architecture suitable for technical evaluation by an F1 team.

## Scope

* Surrogate aerodynamic modeling (pressure, vorticity, drag/downforce).
* Vortex Lattice Method (VLM) and Panel Method physics service.
* Quantum optimization service using QUBO/QAOA.
* Visual front-end (React/Next.js + Three.js + VTK.js).
* Backend (Node/Express + MongoDB) for data persistence.
* Microservices deployed via Docker and orchestrated by docker-compose.

## Deliverables

* GPU surrogate model training scripts.
* ML inference microservice.
* Quantum optimization microservice.
* VLM/Panel physics microservice.
* MERN backend.
* React dark-mode front-end with 3D visualizations.
* docker-compose environment.

## High-Level Timeline

### Phase 1 – Foundation (2–3 weeks)

* Research aerodynamic datasets.
* Implement base VLM/Panel physics engine.
* Build surrogate network architecture.

### Phase 2 – Microservices (3 weeks)

* Implement ML, quantum, and physics APIs.
* Build Node backend and MongoDB collections.
* Configure Docker containers.

### Phase 3 – GPU Surrogate Model (2–4 weeks)

* Data preprocessing.
* Train PyTorch CUDA model.
* ONNX export.

### Phase 4 – Quantum Integration (2–3 weeks)

* QUBO formulation for aerodynamic optimization.
* Integrate Qiskit-Aer backend.
* Validate outputs via physics service.

### Phase 5 – Front-End (4–6 weeks)

* Dark-mode landing page.
* 3D visualization components.
* Simulation dashboards.

### Phase 6 – Integration + Review (3 weeks)

* Combine services.
* Validate performance.
* Prepare F1-ready demo.
