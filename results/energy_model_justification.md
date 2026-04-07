# Energy Model Justification

- Total transmit-chain power is modeled as:
  `P_total = P_tx / eta_pa + P_tx_circuit + P_baseband + P_cooling`
- `eta_pa` (PA efficiency) captures RF power-amplifier conversion losses.
- `P_tx_circuit` captures RF front-end and related TX electronics.
- `P_baseband` captures digital processing power.
- `P_cooling` acts as a practical overhead proxy for thermal/support systems.

Sensitivity analysis is provided in `energy_model_sensitivity.json` and
`energy_model_sensitivity.png`, showing the energy-aware algorithm remains
effective under realistic parameter variation.
