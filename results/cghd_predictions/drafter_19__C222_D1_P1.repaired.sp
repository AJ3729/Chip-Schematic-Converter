* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Diode both_on=n2
D1 n9 n7 Ddefault
L1 n2 n4 1m
C1 n2 n8 1u
* SAME_NODE_SKIPPED Diode both_on=n2
L2 n3 n2 1m
L3 n6 n2 1m
C2 n2 n8 1u
I1 0 n3 DC 1m
D2 n9 n2 Ddefault
L4 n5 n2 1m
D3 n9 n2 Ddefault
Q1 n2 n2 n2 QNPNdefault
I2 0 n5 DC 1m
* SAME_NODE_SKIPPED Diode both_on=n2
C3 n2 n7 1u
C4 n2 n7 1u
I3 0 n6 DC 1m
C5 n2 n8 1u
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end
