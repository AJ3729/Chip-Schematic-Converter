* Auto-generated SPICE netlist (NO TEXT OCR USED)

L1 n1 n3 1m
L2 n3 0 1m
D1 n2 0 Ddefault
D2 0 n1 Ddefault
D3 n2 n3 Zdefault
D4 0 n2 Ddefault
* SAME_NODE_SKIPPED Diode both_on=0
D5 n1 0 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end
