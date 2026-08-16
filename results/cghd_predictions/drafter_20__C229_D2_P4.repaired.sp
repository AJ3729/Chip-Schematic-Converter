* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n4 Ddefault
R1 n4 0 1k
R2 0 n7 1k
R3 0 n5 1k
R4 0 n6 1k
D2 n4 0 Zdefault
R5 n1 n4 1k
Q1 0 n6 n7 QNPNdefault
R6 n3 0 1k
Q2 0 0 0 QNPNdefault
R7 n2 0 1k
Q3 n4 0 0 QNPNdefault
* SAME_NODE_SKIPPED Zener Diode both_on=n4
L1 n3 0 1m
* UNSNAPPED BJT-PNP raw_nodes=[5, None, None]
* SAME_NODE_SKIPPED Diode both_on=0
L2 n2 0 1m
.model Ddefault D
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
