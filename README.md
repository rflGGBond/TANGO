# TANGO-CIQ: Topology-Adaptive Neighbor-Governed Orchestration with Coordinated Inquiry

**From Advisor to Decision-Maker: A Negotiation-Based Multi-Agent Framework for Cooperative Coevolution**

TANGO-CIQ transforms LLM agents from passive advisors into active decision-makers in evolutionary algorithms. Builds upon HMACE (Li et al., 2026) and targets the Influence Blocking Maximization (IBM) problem.

## Core Innovation

Existing LLM-EA methods treat agents as external consultants — they observe, suggest, and wait for the EA engine to accept or reject. TANGO-CIQ upgrades agents to **negotiating decision-makers** that:

1. **Topology-Adaptive Communication Graph**: Dynamically build agent communication topology based on cross-community propagation strength
2. **NR-CIQ (Neighbor-Restricted Coordinated Inquiry)**: Structured JSON-based queries between neighbor agents only
3. **Negotiation-Based Coordination**: Community Bidding Agents negotiate boundary nodes and budgets through iterative proposal/counter-proposal
4. **Multi-Level Verification**: Strategic Reviewer ensures all negotiated decisions satisfy legality, quality, and global optimality constraints

## Installation

```bash
git clone https://github.com/rflGGBond/TANGO-CIQ.git
cd TANGO-CIQ
pip install -r requirements.txt
```

## Quick Start

```bash
python tango_ciq/run.py --graphs congress-Twitter --total_budget 20 110 200 --llm_provider local --llm_model Qwen2.5-7B-Instruct
```

## Repository Structure

```
tango_ciq/
├── run.py                 # Main entry point
├── agents/                # Agent implementations
│   ├── community_bidding_agent.py  # P2P negotiating agents
│   ├── negotiation_coordinator.py  # Global resource optimizer
│   └── strategic_reviewer.py       # Multi-level verifier
├── communication/         # NR-CIQ communication layer
│   ├── topology_graph.py  # Adaptive communication topology
│   ├── nr_ciq.py          # Neighbor-restricted query protocol
│   └── manager.py         # Communication manager
├── core/                  # Evolution engine (adapted from HMACE)
├── environment/           # Negotiation-aware environment
└── utils/                 # Types, LLM client, utilities
```

## Citation

```bibtex
@article{tango-ciq,
  title={TANGO-CIQ: From Advisor to Decision-Maker — A Negotiation-Based Multi-Agent Framework for Cooperative Coevolution},
  author={Li, Fan-Rong and ...},
  year={2026}
}
```

## License

MIT
