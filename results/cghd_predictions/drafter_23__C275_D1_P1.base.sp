* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n9 0 0 0 100k
E2 n7 0 0 n8 100k
E3 n4 0 0 0 100k
C1 n8 n7 1u
C2 n6 0 1u
C3 0 n3 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
C4 n7 0 1u
C5 n8 0 1u
C6 n10 n9 1u
C7 n5 0 1u
R1 n1 n4 1k
C8 n2 0 1u
C9 n9 0 1u
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
C10 0 n10 1u
Q1 n1 n1 n2 QNPNdefault
D1 0 n4 Ddefault
C11 n8 n7 1u
.model Ddefault D
.model QNPNdefault NPN

.op
.end
