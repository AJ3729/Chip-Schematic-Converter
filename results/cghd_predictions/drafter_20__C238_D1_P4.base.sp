* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n3 0 n2 n5 100k
E2 n2 0 n1 n2 100k
E3 n2 0 n13 n2 100k
R1 n3 n1 1k
R2 n12 n9 1k
R3 n5 n3 1k
L1 n7 n6 1m
C1 n1 n2 1u
R4 n5 n2 1k
D1 n15 n9 Ddefault
R5 n11 n14 1k
E4 n7 0 n2 n8 100k
R6 n10 n13 1k
R7 n13 0 1k
R8 n2 n11 1k
I1 n12 n15 AC 1m
R9 n14 n16 1k
V1 n9 n4 DC 5
R10 n11 n14 1k
* UNSNAPPED Resistor raw_nodes=[None, None]
.model Ddefault D

.op
.end
