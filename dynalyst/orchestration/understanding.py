"""How the bifurcation sub-agent maps range requests to scan arguments: a glossary and
worked examples (few-shot), shown in its prompt. Words in these texts are examples of what
users say, never required trigger words.
"""

RANGE_GLOSSARY = """
Scan arguments (what each one means):
- bif_param: the parameter each diagram continues in. It is swept, so it is never also in
  fixed_params, vary_only or param_windows. Left out when the task names none.
- y_species: the species plotted; left out when the task names none.
- fixed_params: parameters held, not sampled: '["X"]' at the model's value, '{"X": 5}' at 5.
- vary_only: the only parameters sampled; every other one is held. A task that names what to
  sample keeps it in every scan.
- param_windows: one parameter's own range: '{"X": {"fold": 3}}', '{"X": {"rel": 0.1}}',
  '{"X": {"abs": [10, 30]}}', '{"X": {"offset": 5}}'. The others still vary as asked.
- rel_window: the relative range of every other sampled parameter (0.05 = 5%);
  spread_factor: the same as a factor.
- baseline_values: a new value a parameter is sampled around; manual_bounds: exact bounds.
- param_min, param_max: the continuation range of bif_param.
- n_samples: how many parameter sets; separate: one plot per category; goal: the behaviour
  looked for (bistability, branch point, Hopf, ultrasensitivity).
"""

RANGE_EXAMPLES = """
Worked examples (p1, p2 ... are the request's model's own parameter names):
1. "vary p1 by a factor of 3, keep p2 and p3 fixed, set p4 to 12, everything else +-1% of its
   allowed range" -> param_windows '{"p1": {"fold": 3}}', fixed_params '["p2", "p3"]',
   baseline_values '{"p4": 12}', rel_window 0.01. NOT vary_only: the others still vary.
2. "vary only p1 by a factor of 3" = "keep every parameter as published except p1, which spans
   a 3-fold range" -> vary_only "p1", param_windows '{"p1": {"fold": 3}}'.
3. "hold everything except p1 and p2, which may change 20%" -> vary_only "p1,p2", rel_window 0.2.
4. "sample within one percent, keep the bifurcation parameter constant" -> rel_window 0.01 only.
5. "20 random sets of model.xml around its values, drawn in p between 1.5 and 150" -> model
   'model.xml', n_samples 20, bif_param 'p', param_min 1.5, param_max 150; nothing else.
6. "10 points of model.xml in p, p5 fixed at 100, the rest within 5%" -> bif_param 'p',
   fixed_params '{"p5": 100}', rel_window 0.05, n_samples 10.
7. "how does Y respond to p" for model.xml -> ONE diagram: run_bifurcation_analysis(param_name
   p, plot_state Y).
8. "run a bifurcation analysis of model.xml", no parameter named -> leave param_name out.
"""
