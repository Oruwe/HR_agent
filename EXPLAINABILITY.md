# HR Talent Evaluator Explainability

## Decision Reasoning
The agent calculates candidate scores by comparing extracted architectural competencies against deterministic role schemas. It decides whether to recommend advancement by evaluating technical trade-off reasoning during live conversational screening interviews.

## Data Inputs
The primary data sources are parsed candidate resumes and real-time audio streams ingested through LiveKit WebRTC. Additionally, it queries pre-configured company competency standards stored as semantic graphs.

## Known Limitations
One major constraint is that the agent cannot evaluate visual whiteboard diagrams in real time. Another known issue is that heavily noisy audio streams may reduce transcription fidelity before VAD confirmation.
