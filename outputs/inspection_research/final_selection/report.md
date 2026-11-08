# Locked validation selection and fresh fixed-policy test

11 candidates on ten common validation sequences 10240-10249; choices locked before test24900-24949.
Selected neural: residual_dispatch_s0. Selected CEM: cem_s1. Overall validation winner: retuned_setup_spt.
This evaluates single fixed selected policies on50 fresh three-day scenarios, not an algorithm average over training seeds.
All service choices use disclosed public rules; inspection benefit is not tested. Normalised cost is not accuracy or money savings.

|policy|full_objective|completion_rate|on_time_rate|inspection_count|breakdowns|belief_coverage90|
|---|---|---|---|---|---|---|
|retuned_setup_spt|36.3081|0.988667|0.601333|0|9.72|0.917417|
|selected_cem|34.6688|0.988667|0.618444|0|7.92|0.916375|
|selected_neural|39.733|0.984222|0.492|0|10.08|0.910708|
|spt|37.7723|0.985111|0.603778|0|6.8|0.9195|

|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|selected_neural|selected_cem|5.06418|2.76889|7.43377|0|50|paired scenario bootstrap for fixed validation-selected policies; no across-training-seed uncertainty|
|selected_neural|retuned_setup_spt|3.4249|1.87464|5.35655|0|50|paired scenario bootstrap for fixed validation-selected policies; no across-training-seed uncertainty|
|selected_neural|spt|1.96073|-0.497098|4.58423|0|50|paired scenario bootstrap for fixed validation-selected policies; no across-training-seed uncertainty|

Selection uncertainty and broader training-seed variation are not captured by these fixed-policy scenario intervals.
All earlier studies and negative comparisons remain part of the research evidence. No test result selects the model.
