* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n2 0 AC 1
V2 n1 0 AC 1
* SAME_NODE_SKIPPED V-AC both_on=0
M1 0 n3 0 0 NMOSdefault
C1 n4 0 1u
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end
