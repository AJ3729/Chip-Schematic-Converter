* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n1 n2 0 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=0
V1 n1 0 DC 5
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
