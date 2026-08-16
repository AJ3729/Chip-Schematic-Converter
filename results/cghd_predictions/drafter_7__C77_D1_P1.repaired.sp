* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n1 n3 QNPNdefault
R1 n1 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
Q2 n3 0 0 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Diode both_on=0
R2 0 n1 1k
R3 n2 n3 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
