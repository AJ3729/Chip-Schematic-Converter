* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n6 0 n10 n12 100k
E2 n1 0 n6 n11 100k
C1 n6 0 1u
R1 n10 n6 1k
R2 n6 n11 1k
* SAME_NODE_SKIPPED Capacitor both_on=n1
R3 n11 n1 1k
R4 n11 n1 1k
R5 n12 0 1k
R6 n2 n1 1k
R7 n6 0 1k
R8 n5 n6 1k
C2 n13 0 1u
R9 n4 n8 1k
R10 n10 n15 1k
V1 n9 0 DC 5
C3 n6 n5 1u
E3 n16 0 n15 n3 100k
C4 n2 n7 1u
C5 n7 n11 1u
R11 n14 n13 1k
R12 n2 n1 1k

.op
.end
