# Independent model-regime confirmation



Two regimes were fixed after exploration and evaluated on five fresh validation scenarios. The fast/low-cost regime is a hypothesis from the exploratory map; default inspection is a contrast.

Both use a wider [0.2,1] initial-health prior than the main policy experiment. Neither is a claim about measured industrial inspection economics.

Nested intervals resample scenarios and matched latent worlds. Each is 97.5%, chosen to allow two simultaneous inspection-versus-alternative comparisons within a regime/headroom pair. They do not cover all regime/headroom comparisons, forecast error or simulator error.

|Regime|Extra peak headroom|Inspection minus alternative|Mean|97.5% interval|

|---|---:|---|---:|---|

|fast_low_cost|0|process|0.0273|[0.0116, 0.0452]|

|fast_low_cost|0|maintain|-0.0123|[-0.0460, 0.0200]|

|fast_low_cost|60|process|0.0282|[0.0138, 0.0456]|

|fast_low_cost|60|maintain|-0.0210|[-0.0654, 0.0193]|

|default_inspection|0|process|0.1234|[0.0985, 0.1505]|

|default_inspection|0|maintain|0.0837|[0.0500, 0.1117]|

|default_inspection|60|process|0.1175|[0.0959, 0.1446]|

|default_inspection|60|maintain|0.0683|[0.0325, 0.1012]|



These are 180-minute model-based incremental-cost comparisons with no residual value. A supported effect here is not evidence that PPO learned it or that the full policy beats rules. Read alongside the full-cycle held-out policy evaluation and negative results.
