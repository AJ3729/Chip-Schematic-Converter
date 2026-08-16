* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n12 1u
C2 0 n5 1u
R1 n12 n5 1k
R2 n7 n12 1k
R3 n1 n4 1k
R4 n3 n6 1k
R5 n1 0 1k
R6 n2 n5 1k
D1 n8 n5 Zdefault
D2 n4 n10 Ddefault
C3 n5 n6 1u
D3 n9 n11 Zdefault
D4 n9 n11 Ddefault
V1 n6 0 DC 5
C4 n4 0 1u
* SAME_NODE_SKIPPED Zener Diode both_on=0
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end
