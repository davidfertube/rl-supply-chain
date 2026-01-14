---
title: RL Supply Chain
emoji: 📦
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - reinforcement-learning
  - supply-chain
  - optimization
  - ppo
---

# RL Supply Chain 📦

**Reinforcement Learning for Supply Chain Optimization**

## 🎯 Purpose

Optimize inventory management using Proximal Policy Optimization (PPO):
- Minimize holding costs
- Reduce stockouts
- Optimize order quantities
- Handle demand uncertainty

## 🤖 Model

- **Algorithm**: PPO (Stable-Baselines3)
- **State**: Inventory levels, demand forecast, lead time
- **Action**: Order quantity
- **Reward**: Negative cost (holding + stockout + ordering)

## 👤 Author

**David Fernandez** | Industrial AI Engineer
