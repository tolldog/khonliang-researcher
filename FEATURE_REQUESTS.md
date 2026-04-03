# Feature Requests from Research Distillation

Generated from analysis of 168 papers on multi-agent LLM systems.
Split into khonliang (library) and autostock (application) requests.

---

## Khonliang Feature Requests

### KH-1: Outcome Feedback on ConsensusEngine

**Papers**: MAGRPO, C3 (Counterfactual Credit Assignment), CoMAS, MARL-PB

**What**: Add `record_outcome(consensus_id, outcome_score, metadata)` to ConsensusEngine. Stores which votes contributed to good/bad outcomes, enabling weight learning.

**API**:
```python
engine.record_outcome(consensus_id="abc123", outcome=0.05, metadata={"symbol": "TSLA"})
history = engine.get_outcome_history(agent_id="quant", limit=100)
```

**Storage**: SQLite table `consensus_outcomes` (consensus_id, votes_json, action, confidence, outcome_score, timestamp).

**Why**: The MAGRPO paper's core finding — joint reward beats independent optimization. C3 adds counterfactual credit: "if we removed this agent's vote, would the outcome change?" Currently ConsensusEngine is fire-and-forget with no feedback loop.

---

### KH-2: Adaptive Weight Learning

**Papers**: C3, Shapley-inspired causal influence (Lazy Agents paper), MARL-PB

**What**: Add `suggest_weights(min_samples)` that analyzes outcome history and returns updated agent weights using leave-one-out counterfactual credit.

**API**:
```python
suggested = engine.suggest_weights(min_samples=30)
# Returns: {"quant": 0.25, "risk": 0.18, ...} or None if insufficient data
# Does NOT auto-apply — human reviews first
```

**Algorithm**: For each agent, compute: "if we removed this agent's vote from past consensuses, how often would the outcome have been worse?" Agents whose votes flip outcomes toward profitable results get higher weight. Also reward non-consensus high-conviction correct votes (from consensus-seeking paper).

**Why**: Currently `DEFAULT_AGENT_WEIGHTS` are hardcoded. The Lazy Agents paper found that some agents contribute nothing — their votes are redundant. Weight learning identifies and downweights lazy agents automatically.

---

### KH-3: Group Sampling on AgentTeam

**Papers**: MAGRPO (GRPO), More Agents Is All You Need (Agent Forest), TUMIX

**What**: Add `sample_count` parameter to `AgentTeam.evaluate()`. Each agent generates N candidate votes; best one is submitted to consensus.

**API**:
```python
result = await team.evaluate(subject, context, sample_count=3)
# Each agent generates 3 votes, keeps the most internally consistent one
```

**Selection heuristic**: Among N samples, pick the one whose action matches the plurality of that agent's samples and has highest confidence. Discards outlier/confused responses.

**Why**: Agent Forest showed that simple sampling + voting improves performance across all model sizes. TUMIX confirmed this with 3.55% accuracy improvement. Cheap with small models (3B/7B).

---

### KH-4: Heuristic Pool (already planned P5.1)

**Papers**: TwinMarket (BDI), Enhancing Reasoning with Collaboration and Memory

**What**: `khonliang.training.HeuristicPool` — extracts reusable rules from outcome history, builds prompt context from them.

**API**:
```python
pool.record_outcome(context={"symbol": "TSLA", "rsi": 28}, decision="BUY", outcome=0.05)
heuristics = pool.extract(min_samples=10, min_confidence=0.6)
context = pool.build_prompt_context(current_context, max_heuristics=5)
# Returns: "When RSI<30 + MACD bullish on tech: 80% success (15 samples)"
```

**Why**: The Thought Communication paper shows agents benefit from persistent learned knowledge. Currently agents start fresh each evaluation. The heuristic pool makes learning persistent across sessions.

---

### KH-5: Blackboard History + Persistence

**Papers**: G-Memory, Memoria, GAAMA, Nemori

**What**: Optional SQLite persistence for Blackboard entries. Add `history()` for expired entries and `query()` with predicates.

**API**:
```python
board = Blackboard(persist_to="data/blackboard.db")  # opt-in persistence
board.history(section="signals", key="TSLA", limit=50)  # includes expired
board.query(predicate=lambda e: e.content.get("action") == "BUY")
```

**Why**: 6 memory papers confirm that agent memory persistence improves performance. Current Blackboard is in-memory only — expired entries are lost. History enables "what blackboard patterns preceded good outcomes?" analysis.

---

### KH-6: Vote Validation

**Papers**: Why Do Multi-Agent LLM Systems Fail? (14 failure modes)

**What**: `ConsensusEngine.validate_votes(votes)` — pre-consensus check that reasoning matches action, flags mismatches.

**API**:
```python
issues = engine.validate_votes(votes)
# Returns: [ValidationIssue(agent="quant", type="reasoning_action_mismatch", 
#           detail="reasoning says 'bearish' but action is BUY")]
```

**Why**: The failure modes paper found "reasoning-action mismatch" is a top failure mode — agents say one thing in reasoning but vote differently. 0.77 kappa for AI-only verification vs 0.88 for human. Cheap check (regex or small-model) catches the most common multi-agent failure.

---

### KH-7: Embedding-Aware Blackboard

**Papers**: Thought Communication, Communicating Activations, Federation of Agents (VCVs)

**What**: Optional `embedding` field on BlackboardEntry. Adds `search_similar()` and `similarity()` methods.

**API**:
```python
board.post(agent_id="quant", section="signals", key="TSLA",
           content="RSI oversold", embedding=[0.12, -0.34, ...])

# Semantic search across all sections
results = board.search_similar(embedding=[...], threshold=0.7, limit=5)

# Compare two entries
sim = board.similarity(key_a="TSLA:quant", key_b="TSLA:sentiment")  # → 0.89
```

**Why**: Communicating Activations paper showed 27% improvement over text-only communication. Federation of Agents uses capability vectors for routing. This is the lightweight version — FastEmbed (CPU, <5ms) produces embeddings alongside text, enabling semantic retrieval and agreement detection without changing the communication protocol.

---

### KH-8: Capability-Based Agent Discovery

**Papers**: Federation of Agents (VCVs + HNSW), MAS² (self-configuring), AgentNet

**What**: Agents register capability vectors. Tasks route to agents by embedding similarity instead of static rules.

**API**:
```python
registry.register_capability(
    agent_id="quant", 
    capability="technical analysis of equity price patterns",
    embedding=[...]  # auto-computed if omitted
)

# Find best agents for a task
matches = registry.find_capable(
    task="analyze unusual options volume on TSLA",
    threshold=0.6, limit=3
)
# Returns: [("quant", 0.85), ("sentiment", 0.72)]
```

**Why**: Currently autostock uses static `DEFAULT_ACTIVATION_RULES` mapping event types to agents. FoA showed 13x improvement with dynamic capability routing. MAS² showed agents can self-configure which tasks they handle.

---

### KH-9: Debate on Disagreement

**Papers**: D3 (Debate Deliberate Decide), Multi-Agent Debate with Adaptive Stability, Talk Isn't Always Cheap, Improving Factuality through Multiagent Debate

**What**: Wire existing `DebateOrchestrator` to auto-trigger when ConsensusEngine detects high disagreement. Add stability detection to know when to stop debating.

**API**:
```python
engine = ConsensusEngine(
    debate_threshold=0.3,  # trigger debate when top 2 actions within 0.3
    debate_orchestrator=orchestrator,
    max_debate_rounds=3,
    stability_detector=AdaptiveStabilityDetector(),  # stop when votes stabilize
)
```

**Why**: D3 found that adversarial debate improves evaluation reliability. But "Talk Isn't Always Cheap" warns that debate can *reduce* quality if not controlled. Adaptive stability detection (from Multi-Agent Debate paper) knows when to stop — when votes stop changing between rounds.

---

## AutoStock Feature Requests

### AS-1: Trade Outcome → Consensus Feedback Loop

**Papers**: MAGRPO, C3, CoMAS

**Where**: `src/trading/trade_executor.py` → `src/agents/consensus.py`

**What**: After a trade closes, call `consensus_engine.record_outcome()` with P&L. Wire consensus_id through TradeDecision → TradeResult for traceability.

**Depends on**: KH-1

---

### AS-2: Periodic Weight Review Task

**Papers**: C3 (LOO credit), Lazy Agents (Shapley)

**Where**: New scheduler task in `src/agent/tasks.py`

**What**: Weekly task calls `consensus_engine.suggest_weights()`, posts suggestion to blackboard + Mattermost for human review. Does NOT auto-apply.

**Depends on**: KH-2

---

### AS-3: Triple Store Cooperation Patterns

**Papers**: TwinMarket (BDI), Thought Communication

**Where**: `src/agents/team.py` after consensus

**What**: Record inter-agent agreement patterns as triples. Query in agent `analyze()` for context.

```python
# After profitable trade where quant + sentiment agreed
triple_store.add("quant+sentiment", "profitable_agreement_on", "TSLA", confidence=0.8)
```

**Depends on**: Existing TripleStore (no new khonliang features needed)

---

### AS-4: Dynamic Activation from Triples

**Papers**: Federation of Agents, MAS²

**Where**: `src/agents/activation.py`

**What**: Before evaluating activation rules, query triples for learned agent-value patterns. Suppress agents that don't add value for this event type.

**Depends on**: AS-3

---

### AS-5: Group Sampling for High-Stakes Decisions

**Papers**: MAGRPO, More Agents Is All You Need, TUMIX

**Where**: `src/agents/team.py`

**What**: When position size > 10% of portfolio, use `sample_count=3` for more robust consensus.

**Depends on**: KH-3

---

### AS-6: Outcome-Derived Knowledge Ingestion

**Papers**: Enhancing Reasoning with Collaboration and Memory, G-Memory

**Where**: New weekly scheduler task

**What**: Query consensus outcomes for patterns, extract heuristics, ingest as Tier 3 knowledge. Agents pick up via `knowledge_context()`.

**Depends on**: KH-4 (HeuristicPool)

---

### AS-7: Debate on Split-Consensus Anomaly Signals

**Papers**: D3, Multi-Agent Debate with Adaptive Stability

**Where**: `src/agents/watcher_notifications.py`

**What**: When MarketWatcher fires an anomaly AND consensus is split (no clear majority), trigger debate between disagreeing agents before final decision.

**Depends on**: KH-9

---

### AS-8: Belief Shift Tracking

**Papers**: TwinMarket (BDI framework)

**Where**: `src/agents/team.py`

**What**: Track when an agent's vote flips across consecutive evaluations of the same symbol. Record as triples. Belief shifts are early signals.

```python
triple_store.add("quant", "belief_shift", "TSLA:HOLD→BUY", confidence=0.8)
```

**Depends on**: AS-3

---

### AS-9: Training Data Collection

**Papers**: MALT, Chain-of-Agents

**Where**: New module `src/training/collector.py`

**What**: Store group samples + outcomes as training data (SQLite), even before fine-tuning infrastructure exists. MALT showed offline data collection → DPO training works on 1.5B-8B models.

**Depends on**: KH-3 (group sampling provides the data)

---

### AS-10: Embedding-Enhanced Votes

**Papers**: Communicating Activations, Thought Communication

**Where**: `src/agents/models.py` AgentVote

**What**: Add optional `context_vector` field to AgentVote. ConsensusEngine uses it for reasoning-action mismatch detection and agreement clustering.

```python
@dataclass
class AgentVote:
    agent_id: str
    action: AgentAction
    confidence: float
    reasoning: str
    factors: Dict[str, float]
    context_vector: Optional[List[float]] = None  # ← new
```

**Depends on**: KH-7 (embedding-aware Blackboard provides the infrastructure)

---

### AS-11: MCP Tool Discovery for Agent Actions

**Papers**: FinMCP-Bench, Semantic Tool Discovery, TheMCPCompany, MCP Design Choices

**Where**: `src/agents/base.py` or new `src/agents/tools.py`

**What**: Agents discover available actions via MCP tool registry instead of hardcoded methods. Each agent's `analyze()` can call MCP tools dynamically based on what's available.

**Depends on**: Existing khonliang MCP server + new tool registration pattern

---

## Dependency Graph

```
KH-1 (outcome recording)
 ├── AS-1 (wire trade outcomes)
 ├── KH-2 (weight learning)
 │    └── AS-2 (weight review task)
 └── AS-3 (triple patterns)
      ├── AS-4 (dynamic activation)
      └── AS-8 (belief shift tracking)

KH-3 (group sampling)
 ├── AS-5 (high-stakes sampling)
 └── AS-9 (training data collection)

KH-4 (heuristic pool)
 └── AS-6 (knowledge ingestion)

KH-5 (blackboard history)

KH-6 (vote validation)

KH-7 (embedding blackboard)
 ├── KH-8 (capability discovery)
 │    └── AS-4 (replaces static activation)
 └── AS-10 (embedding votes)

KH-9 (debate on disagreement)
 └── AS-7 (split-consensus debate)

AS-11 (MCP tool discovery) ← independent
```

## Implementation Priority

**Phase 1 — Close the feedback loop** (highest value, foundational):
  KH-1 → AS-1 → AS-3 → KH-6

**Phase 2 — Learn from outcomes**:
  KH-2 → AS-2 → KH-4 → AS-6

**Phase 3 — Improve consensus quality**:
  KH-3 → AS-5 → KH-9 → AS-7

**Phase 4 — Semantic communication**:
  KH-7 → KH-8 → AS-10 → AS-4

**Phase 5 — Training pipeline**:
  AS-9 → AS-11

**Start with**: KH-1 → AS-1 → AS-3. This is the minimum viable feedback loop.
