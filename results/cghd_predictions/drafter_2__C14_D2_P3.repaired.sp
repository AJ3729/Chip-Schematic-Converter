* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n2 0 DC 5
D1 0 n1 Ddefault
I1 n3 0 DC 1m
* SAME_NODE_SKIPPED Diode both_on=0
* SAME_NODE_SKIPPED Zener Diode both_on=0
M1 0 0 0 0 NMOSdefault
* SAME_NODE_SKIPPED Diode both_on=0
.model Ddefault D
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
