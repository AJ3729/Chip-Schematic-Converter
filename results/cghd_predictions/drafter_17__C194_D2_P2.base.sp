* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n5 0 n10 n11 100k
E2 n1 0 n5 n9 100k
R1 n5 n9 1k
R2 n10 n5 1k
R3 n9 n1 1k
R4 n9 n1 1k
R5 n5 0 1k
R6 n6 n1 1k
R7 n11 0 1k
R8 n10 0 1k
R9 n3 n1 1k
C1 n1 n12 1u
C2 n5 0 1u
C3 n13 0 1u
R10 n3 n1 1k
R11 n4 n5 1k
R12 n14 n13 1k
V1 n12 0 DC 5
C4 n2 n4 1u
C5 n9 n8 1u
L1 n3 n7 1m
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
C6 n5 n4 1u

.op
.end
