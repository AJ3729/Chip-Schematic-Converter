* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n5 0 Ddefault
Q1 n8 n7 n5 QNPNdefault
Q2 n2 n2 n3 QNPNdefault
D2 n5 0 Ddefault
R1 n1 n2 1k
R2 n4 n8 1k
* SAME_NODE_SKIPPED Zener Diode both_on=0
R3 n4 n7 1k
R4 n2 n3 1k
R5 0 n4 1k
R6 n1 0 1k
R7 n1 n2 1k
D3 n3 n2 Zdefault
Q3 n2 0 n4 QNPNdefault
Q4 n6 n4 n3 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n6 n6 0 1e+09

.op
.end
