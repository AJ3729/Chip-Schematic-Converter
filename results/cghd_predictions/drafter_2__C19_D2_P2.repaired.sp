* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 0 0 0 0 NMOSdefault
* SAME_NODE_SKIPPED V-DC both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED I-AC both_on=0
* SAME_NODE_SKIPPED Diode both_on=0
I1 n2 0 DC 1m
* SAME_NODE_SKIPPED V-DC both_on=0
C1 0 n1 1u
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end
