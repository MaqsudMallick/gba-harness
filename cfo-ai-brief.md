# cfo.ai planning brief

Use this as the starting context in cfo.ai. It is a planning input, not a completed cfo.ai
business plan; replace assumptions with measured values from agent traces and user tests.

## Product

An observable game-agent arena where humans and AI agents compete in real games. Developers
can plug in a game-specific action policy and a model provider without changing the emulator
or match runner. The initial Super Mario integration proves the control loop; multiplayer and
agent-vs-human formats are the intended product direction.

## Customers and value

- AI teams: visually evaluate and compare multimodal agents in a live environment.
- Game studios: prototype bots, difficulty levels, play-testing agents, and tournaments.
- Players and creators: host human-vs-agent challenges with visible reasoning and replays.

The differentiator is not another scripted game bot. It is a provider-neutral competition
and evaluation layer with traceable decisions, bounded controls, recovery, and replayable
evidence.

## Pricing hypotheses to model

- Developer: $29/month, including local arena tooling and a small hosted match allowance.
- Team: $199/month, including shared experiments, private leaderboards, and higher limits.
- Usage: charge per hosted agent-minute or match beyond the included allowance.
- Enterprise/studio: annual contract for private games, deployment, and support.

Treat all prices as hypotheses. Ask cfo.ai to compare subscription-only, usage-only, and
hybrid pricing and calculate the break-even match volume for each.

## Cost drivers

- Vertex AI image-input and output-token costs per decision.
- Decisions per match and average match duration.
- Emulator compute per concurrent match.
- Trace ingestion/storage and replay artifact storage.
- Engineering, support, payment processing, and community/tournament operations.

Use measured neatlogs token counts and latency from representative matches. Do not estimate
model cost from the synthetic Doctor trace.

## Scenarios requested from cfo.ai

Build conservative, base, and growth cases for 12 months. Include:

- customers by plan and monthly churn;
- matches per customer and inference cost per match;
- monthly recurring revenue, gross profit, and gross margin;
- fixed versus variable costs;
- cash burn, runway, and break-even month;
- sensitivity to model cost, match length, and conversion rate.

## Evidence still needed

- At least one complete real agent match with neatlogs token and latency measurements.
- Vertex cost per decision and emulator compute per agent-minute.
- Five user interviews or comparable demand signals.
- cfo.ai shareable business-plan output for the neatHack submission.
