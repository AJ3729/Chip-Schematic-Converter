* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n2 0 0 QNPNdefault
C1 n1 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
Q2 0 n1 n2 QNPNdefault
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
