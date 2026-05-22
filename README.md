# UAV-CodeAgents
[UAV-CodeAgents: https://arxiv.org/pdf/2505.07236](https://arxiv.org/pdf/2505.07236) inspired agent framework
@misc{sautenkov2025uavcodeagentsscalableuavmission,
      title={UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning}, 
      author={Oleg Sautenkov and Yasheerah Yaqoot and Muhammad Ahsan Mustafa and Faryal Batool and Jeffrin Sam and Artem Lykov and Chih-Yung Wen and Dzmitry Tsetserukou},
      year={2025},
      eprint={2505.07236},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.07236}, 
}
---

## Organization Summary

| **Section** | **Key Classes** | **Responsibility** |
|---|---|---|
| **Domain** | `Frame`, `SemanticAnnotation`, `Waypoint`, `FrameEvidence`, `RetrievedEvidence`, `QueryPlan` | Raw data structures and enriched evidence |
| **MCP Protocol** | `MCPContext`, `MCPResult`, `MCPTool` | Universal plugin contract |
| **Perception** | `PerceptionService`, `FrameEnrichmentService`, `EvidenceBuilder` | VLM, detection, enrichment |
| **Retrieval/Index** | `HybridRetriever`, `EvidenceIndexer`, `RetrievalPlugin`, `TemporalReasoner`, `SpatialReasoner` | Search and index via MCP |
| **Geospatial** | `GeodnetPlugin` | Geospatial lookups as MCP tool |
| **Reasoning** | `ReActState`, `ReActStep`, `QueryPlanner`, `ThoughtGenerator`, `SufficiencyJudge`, `GroundedAnswerSynthesizer`, `ToolRouter`, `ReActController` | Bounded, stateful ReAct loop with tool routing |
| **Validation** | `JudgeScore`, `JudgeInput`, `JudgeEvaluator`, `ValidationPlugin`, `ValidationPipeline` | LLM-as-a-judge scoring |
| **Coordination** | `MessageBus`, `UAVAgent`, `AirspaceManagerAgent` | Asynchronous UAV orchestration |
| **Integration** | `DroneVideoSystem` | Top-level orchestrator |

## Key Design Principles 

1. **All external capabilities are MCP tools** — retrieval, geospatial, validation
2. **ReAct loop stays pure** — no direct LLM calls, only tool invocations
3. **UAV coordination preserved** — `MessageBus` carries observations; manager delegates reasoning
4. **Fully testable & composable** — each layer has clear contracts
5. **Production-ready structure** — stub methods with `raise NotImplementedError` guide implementation

Copy, paste, and implement the stubs for your specific storage, VLM, LLM, and detector backends.